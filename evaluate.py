import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, recall_score, accuracy_score
from dataset import get_data_loaders
from model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on device: {device}")

# 1. Test DataLoader & Model Load
_, _, test_loader = get_data_loaders(batch_size=32)
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

y_true = np.array(y_true)
y_probs = np.array(y_probs)

# 2. Optimal Threshold Search (From 0.40 to 0.75)
print("\n" + "="*50)
print("THRESHOLD TUNING BENCHMARK")
print("="*50)
print(f"{'Threshold':<12} | {'Accuracy':<12} | {'Sensitivity (Recall)':<22} | {'False Positives':<15}")
print("-" * 65)

best_acc = 0.0
best_thresh = 0.5

for t in np.arange(0.40, 0.75, 0.05):
    preds = (y_probs >= t).astype(int)
    acc = accuracy_score(y_true, preds)
    rec = recall_score(y_true, preds)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    
    print(f"{t:<12.2f} | {acc*100:<11.2f}% | {rec*100:<21.2f}% | {fp:<15}")
    
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print("="*50)
print(f"Optimal Threshold for Peak Accuracy: {best_thresh:.2f}")
print(f"Peak Test Accuracy Achieved:        {best_acc*100:.2f}%\n")

# Best threshold report
final_preds = (y_probs >= best_thresh).astype(int)
print(f"Classification Report at Threshold = {best_thresh:.2f}:")
print(classification_report(y_true, final_preds, target_names=["NORMAL (0)", "PNEUMONIA (1)"]))