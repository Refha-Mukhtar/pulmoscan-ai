import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
from model import build_model

# 1. Device check (GPU available hai toh CUDA, nahi toh CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Hyperparameters
EPOCHS = 3          # Shuruat ke liye 3 epochs
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# 3. Load Data & Model
print("Loading data...")
train_loader, val_loader, _ = get_data_loaders(batch_size=BATCH_SIZE)

model = build_model().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# 4. Training Loop
best_val_loss = float('inf')

print("\nStarting Training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Har 40 batches ke baad progress print karein
        if (batch_idx + 1) % 40 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Step [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

    train_loss = running_loss / total
    train_acc = correct / total

    # Validation Phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    print(f"\n--- Epoch {epoch+1} Summary ---")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%\n")

    # Best model save karein
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("--> Best model saved as 'best_model.pth'!\n")

print("Training Complete!")