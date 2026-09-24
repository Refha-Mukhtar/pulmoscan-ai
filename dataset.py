import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

# 1. Medical Image Augmentations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_data_loaders(data_dir="./data", batch_size=32):
    # Datasets load karein
    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(f"{data_dir}/val", transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(f"{data_dir}/test", transform=val_test_transforms)

    # 2. Imbalance Handle karne ke liye WeightedRandomSampler
    targets = train_dataset.targets
    # Count: Normal = 1341, Pneumonia = 3875
    class_counts = [1341, 3875]
    class_weights = [1.0 / c for c in class_counts]
    sample_weights = [class_weights[label] for label in targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # 3. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders()
    print("DataLoader successfully ready!")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")