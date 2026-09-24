import torch
import torch.nn as nn
from torchvision import models

def build_model(pretrained=True):
    # Pre-trained ResNet-18 load karein (lightweight & fast for laptops)
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Backbone freeze karein (initial training me pehle layers train nahi hongi)
    for param in model.parameters():
        param.requires_grad = False

    # Final Fully Connected (fc) layer ko medical binary output ke liye replace karein
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 1)  # Binary output: 1 logit
    )
    
    return model

if __name__ == "__main__":
    model = build_model()
    print("Model successfully built!")
    
    # Test forward pass with dummy image batch (batch of 2 images, 3 channels, 224x224)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape} (Expected: torch.Size([2, 1]))")