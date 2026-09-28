# 🫁 PulmoScan AI: Enterprise Clinical Decision Support System

An end-to-end medical deep learning and explainable AI (XAI) workstation for **Pneumonia detection and triage** from Anterior-Posterior Chest Radiographs. Built with **PyTorch**, **ResNet-18**, and **Streamlit**.

---

## 🌟 Key Features

- **Class-Imbalance Mitigation:** Real-world medical datasets suffer from heavy class imbalance (~3:1 Pneumonia to Normal). Resolved using `WeightedRandomSampler` and inverse frequency weighting.
- **Explainable AI (Grad-CAM):** Custom backpropagation gradient hooks on the final convolutional layer (`layer4[1].conv2`) to visually localize alveolar opacities and lung consolidations.
- **Interactive Split Slider:** Real-time side-by-side transition slider comparing raw radiographs with Grad-CAM salience overlays.
- **3-Tier Clinical Triage Protocol:**
  - `Normal / Clear (< 35%)`
  - `Equivocal / Borderline (35% - 65%)` — Flagged for senior radiologist review
  - `High Suspicion (> 65%)` — Critical alert
- **Emergency Department Batch Queue:** Multi-study parallel ingestion that ranks cases by acuity to fast-track STAT patients.
- **Branded PDF Radiology Reports:** One-click generation of clinical PDF reports complete with patient metadata, findings, and embedded dual-radiograph images.

---

## 📊 Performance Benchmarks (Strict Unseen Test Split)

| Metric | Benchmark Result |
| :--- | :--- |
| **Model Architecture** | ResNet-18 Deep Convolutional Neural Network |
| **Unseen Test Accuracy** | **88.8% (~89.0%)** |
| **Sensitivity / Recall (Pneumonia)** | **96.7%** (377 / 390 detected) |
| **ROC-AUC Score** | **0.957** |
| **Normal Precision** | **93.0%** |
| **Optimal Decision Threshold** | **0.55** |


## 🛠️ Project Structure

```text
medical_diagnosis/
├── data/                      # Chest X-Ray images (train, val, test)
├── dataset.py                 # Augmentation pipeline & WeightedRandomSampler
├── model.py                   # Transfer learning ResNet architecture
├── train.py                   # Training loop with validation checkpointing
├── evaluate.py                # Sensitivity, Recall & Confusion Matrix analysis
├── predict.py                 # Standalone single-study CLI inference
├── gradcam.py                 # Standalone Grad-CAM visualizer
├── app.py                     # Full-featured Streamlit Clinical Console
├── requirements.txt           # Project dependencies
├── .gitignore                 # Excludes heavy datasets and weights
└── README.md                  # Project documentation