import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, recall_score
from dataset import get_data_loaders
from model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on device: {device}")

# 1. Test DataLoader load karein
_, _, test_loader = get_data_loaders(batch_size=32)

# 2. Saved model load karein
model = build_model(pretrained=False).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

y_true = []
y_probs = []

print("Running evaluation on test data...")
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)

        y_probs.extend(probs.cpu().numpy().flatten())
        y_true.extend(labels.numpy().flatten())

# Default threshold 0.5 par predictions
y_pred = [1 if p >= 0.5 else 0 for p in y_probs]

# 3. Metrics Calculate karein
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
recall = recall_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_probs)

print("\n================ EVALUATION REPORT ================")
print(f"Confusion Matrix:")
print(f"  [TN: {tn:<4} | FP: {fp:<4}]  (Healthy correctly diagnosed vs False Alarm)")
print(f"  [FN: {fn:<4} | TP: {tp:<4}]  (Sick missed as Healthy vs Sick correctly diagnosed)")
print("---------------------------------------------------")
print(f"Critical Medical Metric - Recall (Sensitivity): {recall * 100:.2f}%")
print(f"ROC-AUC Score: {auc:.4f}")
print("---------------------------------------------------")
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=["NORMAL (0)", "PNEUMONIA (1)"]))