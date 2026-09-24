import os
import io
import base64
import tempfile
import datetime
import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from model import build_model

# ReportLab for Clinical PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# --- PAGE SETUP ---
st.set_page_config(
    page_title="PulmoScan AI • Enterprise Clinical Suite",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    code, pre { font-family: 'JetBrains Mono', monospace; }
    .stApp { background-color: #0b0f19; }

    .clinical-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.2rem;
    }
    .header-title {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f8fafc 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .diagnostic-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(51, 65, 85, 0.6);
        border-radius: 12px;
        padding: 1.1rem;
        height: 100%;
    }
    .card-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .card-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .triage-high {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
    .triage-borderline {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #f59e0b;
        color: #fbbf24;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
    .triage-normal {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #34d399;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Device & Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_trained_model():
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = True
    return model

model = load_trained_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def compute_inference_and_cam(image, colormap_choice, alpha):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    hook_handle = model.layer4[1].conv2.register_forward_hook(forward_hook)

    orig_resized = image.resize((224, 224))
    img_tensor = transform(image).unsqueeze(0).to(device)

    output = model(img_tensor)
    prob = torch.sigmoid(output).item()

    model.zero_grad()
    output.backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    if cam.max() != 0:
        cam = (cam - np.min(cam)) / np.max(cam)

    cmap_dict = {
        "JET (Standard)": cv2.COLORMAP_JET,
        "TURBO (Modern)": cv2.COLORMAP_TURBO,
        "INFERNO (High Contrast)": cv2.COLORMAP_INFERNO,
        "VIRIDIS (Perceptual)": cv2.COLORMAP_VIRIDIS
    }
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cmap_dict.get(colormap_choice, cv2.COLORMAP_JET))
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    np_orig = np.array(orig_resized)
    superimposed = np.uint8((1.0 - alpha) * np_orig + alpha * heatmap)

    hook_handle.remove()
    return prob, superimposed, heatmap, orig_resized

def get_triage_info(prob, threshold):
    if prob >= max(0.65, threshold):
        return "PNEUMONIA HIGH SUSPICION", "triage-high", "STAT 🚨", "Significant pulmonary opacity / consolidation detected. Immediate clinical workup advised.", "High Probability"
    elif prob <= min(0.35, threshold):
        return "NORMAL / CLEAR STUDY", "triage-normal", "LOW 🟢", "No acute focal airspace consolidation or overt infiltrates identified.", "Low Probability"
    else:
        return "EQUIVOCAL / BORDERLINE", "triage-borderline", "ROUTINE ⚠️", "Indeterminate findings. Mild interstitial markings present. Senior review advised.", "Borderline"

# PDF Generator Function
def generate_clinical_pdf(patient_data, orig_img, overlay_img):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    # Header
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor("#0284c7")
    )
    story.append(Paragraph("<b>PULMOSCAN CLINICAL RADIOLOGY LABORATORY</b>", title_style))
    story.append(Paragraph("<font size=9 color='#64748b'>Automated Chest Radiograph Computer-Assisted Triage System</font>", styles['Normal']))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=15))

    # Patient Metadata Table
    meta_data = [
        [Paragraph(f"<b>Patient ID:</b> {patient_data['id']}", styles['Normal']), Paragraph(f"<b>Date:</b> {patient_data['date']}", styles['Normal'])],
        [Paragraph(f"<b>Age / Sex:</b> {patient_data['age']} Y / {patient_data['sex']}", styles['Normal']), Paragraph(f"<b>Symptoms:</b> {patient_data['symptoms']}", styles['Normal'])],
        [Paragraph(f"<b>Triage Priority:</b> {patient_data['priority']}", styles['Normal']), Paragraph(f"<b>Applied Threshold:</b> {patient_data['threshold']}", styles['Normal'])]
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))

    # Diagnostic Findings Box
    verdict_color = colors.HexColor("#dc2626") if "HIGH" in patient_data['status'] else (colors.HexColor("#d97706") if "EQUIVOCAL" in patient_data['status'] else colors.HexColor("#16a34a"))
    finding_data = [
        [Paragraph(f"<b>DIAGNOSTIC CLASSIFICATION:</b> <font color='{verdict_color.hexval()}'><b>{patient_data['status']}</b></font>", styles['Normal'])],
        [Paragraph(f"<b>Pneumonia Probability:</b> {patient_data['prob']:.2f}% | <b>Model Certainty:</b> {patient_data['confidence']:.2f}%", styles['Normal'])],
        [Paragraph(f"<b>Radiological Interpretation:</b> {patient_data['note']}", styles['Normal'])]
    ]
    finding_table = Table(finding_data, colWidths=[540])
    finding_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1.5, verdict_color),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(finding_table)
    story.append(Spacer(1, 15))

    # Embed Images Side-by-Side
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1, tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
        orig_img.save(f1.name)
        Image.fromarray(overlay_img).save(f2.name)
        
        img_table = Table([
            [Paragraph("<b>Raw Anterior-Posterior Radiograph</b>", styles['Normal']), Paragraph("<b>Grad-CAM Diagnostic Attention Overlay</b>", styles['Normal'])],
            [RLImage(f1.name, width=220, height=220), RLImage(f2.name, width=220, height=220)]
        ], colWidths=[270, 270])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(img_table)

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94a3b8"), spaceAfter=10))
    story.append(Paragraph("<font size=8 color='#64748b'><b>DISCLAIMER:</b> This automated preliminary report was synthesized via Deep Convolutional Neural Network (ResNet-18). Final clinical validation must be corroborated by a board-certified Radiologist.</font>", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# Helper: Interactive Split Slider HTML/JS
def render_image_slider(img_left, img_right):
    def to_b64(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    b64_left = to_b64(img_left)
    b64_right = to_b64(Image.fromarray(img_right))

    slider_html = f"""
    <div style="position:relative; width:100%; max-width:540px; height:460px; margin:auto; overflow:hidden; border-radius:12px; border:2px solid #334155; user-select:none;">
        <img src="data:image/png;base64,{b64_right}" style="position:absolute; top:0; left:0; width:100%; height:100%; object-fit:contain; background:#020617;">
        <div id="slider-mask" style="position:absolute; top:0; left:0; width:50%; height:100%; overflow:hidden; border-right:3px solid #38bdf8; background:#020617;">
            <img src="data:image/png;base64,{b64_left}" style="position:absolute; top:0; left:0; width:540px; height:100%; object-fit:contain;">
        </div>
        <input type="range" min="0" max="100" value="50" style="position:absolute; bottom:15px; left:5%; width:90%; z-index:20; cursor:pointer;" oninput="document.getElementById('slider-mask').style.width = this.value + '%'">
    </div>
    <p style="text-align:center; color:#94a3b8; font-size:0.8rem; margin-top:6px;">↔️ Drag slider horizontally: <b>Left = Raw Radiograph</b> | <b>Right = Grad-CAM Heatmap</b></p>
    """
    st.components.v1.html(slider_html, height=510)

# ==================== NAVIGATION / SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🫁 Navigation Mode")
    app_mode = st.radio("Select Operational Workflow:", ["Single Study Workstation", "Emergency Triage Queue (Batch)"])
    st.markdown("---")

    st.markdown("### ⚙️ Clinical Model Controls")
    decision_threshold = st.slider("Decision Cutoff", min_value=0.20, max_value=0.80, value=0.50, step=0.05)
    alpha_slider = st.slider("Heatmap Opacity", min_value=0.10, max_value=0.90, value=0.45, step=0.05)
    colormap_choice = st.selectbox("Salience Colormap", ["JET (Standard)", "TURBO (Modern)", "INFERNO (High Contrast)", "VIRIDIS (Perceptual)"])
    st.markdown("---")

    if app_mode == "Single Study Workstation":
        st.markdown("### 👤 Patient Information")
        patient_id = st.text_input("Patient ID / MRN", value="PXM-98421")
        patient_age = st.number_input("Age", min_value=1, max_value=110, value=48)
        patient_sex = st.selectbox("Biological Sex", ["Male", "Female", "Other"])
        symptoms = st.multiselect("Symptoms", ["Fever", "Productive Cough", "Dyspnea / SOB", "Pleuritic Chest Pain", "Fatigue"], default=["Fever", "Dyspnea / SOB"])
        st.markdown("---")
        st.markdown("### 🧪 Quick Benchmarks")
        sample_choice = st.radio("Load Pre-set Benchmark:", ["Custom Upload", "Normal Lung (Healthy)", "Pneumonia Case (Pathology)"])

# ==================== HERO BAR ====================
st.markdown("""
<div class="clinical-header">
    <div style="font-size:0.75rem; color:#38bdf8; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;">Hospital Enterprise Deployment • v3.0</div>
    <h1 class="header-title">PulmoScan AI Diagnostic Console</h1>
    <div style="color:#94a3b8; font-size:0.9rem; margin-top:2px;">Real-time ResNet-18 Pneumonia Triage, Interactive Salience Morphing, and Batch Prioritization</div>
</div>
""", unsafe_allow_html=True)

# ==================== WORKFLOW 1: SINGLE WORKSTATION ====================
if app_mode == "Single Study Workstation":
    selected_image = None

    if sample_choice == "Normal Lung (Healthy)":
        normal_dir = "./data/test/NORMAL"
        if os.path.exists(normal_dir) and len(os.listdir(normal_dir)) > 0:
            file_name = [f for f in os.listdir(normal_dir) if not f.startswith('.')][0]
            selected_image = Image.open(os.path.join(normal_dir, file_name)).convert("RGB")
    elif sample_choice == "Pneumonia Case (Pathology)":
        pneumo_dir = "./data/test/PNEUMONIA"
        if os.path.exists(pneumo_dir) and len(os.listdir(pneumo_dir)) > 0:
            file_name = [f for f in os.listdir(pneumo_dir) if not f.startswith('.')][0]
            selected_image = Image.open(os.path.join(pneumo_dir, file_name)).convert("RGB")
    elif sample_choice == "Custom Upload":
        uploaded_file = st.file_uploader("📂 Upload Chest Radiograph (PNG, JPG, JPEG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            selected_image = Image.open(uploaded_file).convert("RGB")

    if selected_image is not None:
        with st.spinner("Analyzing neural activations & backpropagating salience..."):
            prob, superimposed, heatmap, orig_resized = compute_inference_and_cam(selected_image, colormap_choice, alpha_slider)

        status, status_class, priority_badge, clinical_note, severity_level = get_triage_info(prob, decision_threshold)
        confidence = prob if prob >= decision_threshold else (1.0 - prob)

        # KPI Metrics Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class="diagnostic-card"><div class="card-label">Diagnostic Classification</div><div class="{status_class}">{status}</div></div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="diagnostic-card"><div class="card-label">Pathology Probability</div><div class="card-value">{prob * 100:.1f}%</div><div style="font-size:0.75rem; color:#64748b;">Cutoff: {decision_threshold:.2f}</div></div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="diagnostic-card"><div class="card-label">Model Certainty</div><div class="card-value">{confidence * 100:.1f}%</div><div style="font-size:0.75rem; color:#10b981;">ResNet Backbone</div></div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class="diagnostic-card"><div class="card-label">Emergency Priority</div><div class="card-value" style="font-size:1.4rem;">{priority_badge}</div><div style="font-size:0.75rem; color:#64748b;">Triage Protocol</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs
        tab_slider, tab_triple, tab_pdf = st.tabs(["🔀 Interactive Split Slider", "🔬 3-Panel Inspection", "📑 Branded PDF Clinical Report"])

        with tab_slider:
            st.markdown("##### 🫁 Interactive Radiograph vs. Grad-CAM Salience Slider")
            render_image_slider(orig_resized, superimposed)

        with tab_triple:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption("1. RAW RADIOGRAPH")
                st.image(orig_resized, use_container_width=True)
            with c2:
                st.caption("2. SALIENCE HEATMAP")
                st.image(heatmap, use_container_width=True)
            with c3:
                st.caption("3. ANATOMICAL OVERLAY")
                st.image(superimposed, use_container_width=True)

        with tab_pdf:
            st.markdown("##### 📄 Official Radiological Summary Preview")
            patient_meta = {
                "id": patient_id,
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "age": patient_age,
                "sex": patient_sex,
                "symptoms": ', '.join(symptoms) if symptoms else 'None Reported',
                "status": status,
                "prob": prob * 100,
                "confidence": confidence * 100,
                "threshold": decision_threshold,
                "priority": priority_badge,
                "note": clinical_note
            }
            
            pdf_bytes = generate_clinical_pdf(patient_meta, orig_resized, superimposed)
            
            st.info(f"**Impression Summary:** {clinical_note}")
            st.download_button(
                label="⬇️ Download Official Branded Radiology Report (PDF with Images)",
                data=pdf_bytes,
                file_name=f"Clinical_Report_{patient_id}.pdf",
                mime="application/pdf"
            )

# ==================== WORKFLOW 2: BATCH EMERGENCY QUEUE ====================
elif app_mode == "Emergency Triage Queue (Batch)":
    st.markdown("### 🚨 Emergency Department Multi-Patient Triage Queue")
    st.caption("Upload multiple patient scans simultaneously. The system will automatically run parallel inferences, rank cases by acuity, and prioritize emergency patients.")

    batch_files = st.file_uploader("📂 Select Multiple Radiographs", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if batch_files:
        queue_records = []
        progress_bar = st.progress(0)

        for i, file in enumerate(batch_files):
            img = Image.open(file).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.sigmoid(output).item()

            status, _, priority_badge, note, _ = get_triage_info(prob, decision_threshold)

            queue_records.append({
                "Filename": file.name,
                "Probability": prob,
                "Pneumonia Risk": f"{prob * 100:.1f}%",
                "Triage Classification": status,
                "Urgency": priority_badge,
                "Action": "STAT Review" if "HIGH" in status else ("Second Opinion" if "EQUIVOCAL" in status else "Routine Discharge")
            })
            progress_bar.progress((i + 1) / len(batch_files))

        # Sort: High Risk / STAT first
        df_queue = pd.DataFrame(queue_records)
        df_queue = df_queue.sort_values(by="Probability", ascending=False).reset_index(drop=True)
        df_queue.index += 1  # 1-based ranking

        # Summary KPIs
        stat_count = len(df_queue[df_queue["Triage Classification"].str.contains("HIGH")])
        equivocal_count = len(df_queue[df_queue["Triage Classification"].str.contains("EQUIVOCAL")])
        clear_count = len(df_queue[df_queue["Triage Classification"].str.contains("NORMAL")])

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Total Studies Processed", len(df_queue))
        q2.metric("STAT Urgent Cases", stat_count, delta="Immediate Alert" if stat_count > 0 else None, delta_color="inverse")
        q3.metric("Equivocal / Review", equivocal_count)
        q4.metric("Clear / Normal", clear_count)

        st.markdown("#### 📋 Prioritized Triage Register")
        display_df = df_queue.drop(columns=["Probability"])
        st.dataframe(display_df, use_container_width=True)