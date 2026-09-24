import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import build_model

# 1. Device & Model Load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(pretrained=False).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# FIX: Grad-CAM ke liye gradients enable hona zaroori hai
for param in model.parameters():
    param.requires_grad = True

gradients = []
activations = []

def forward_hook(module, input, output):
    activations.append(output)
    # Activation tensor par backward hook register karein
    output.register_hook(lambda grad: gradients.append(grad))

# ResNet18 ke target conv layer par hook lagayein
target_layer = model.layer4[1].conv2
target_layer.register_forward_hook(forward_hook)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def generate_gradcam(image_path, save_output_path="gradcam_result.png"):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} nahi mili!")
        return

    # Image load karein
    orig_img = Image.open(image_path).convert('RGB')
    orig_img_resized = orig_img.resize((224, 224))
    img_tensor = transform(orig_img).unsqueeze(0).to(device)

    # Hooks buffer reset
    gradients.clear()
    activations.clear()

    # Forward pass
    output = model(img_tensor)
    prob = torch.sigmoid(output).item()

    # Backward pass for gradients
    model.zero_grad()
    output.backward()

    # Extract gradients and activations
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))

    # Heatmap calculation
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]

    cam = np.maximum(cam, 0)  # ReLU
    cam = cv2.resize(cam, (224, 224))
    if cam.max() != 0:
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

    # Color heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose on original image
    np_img = np.array(orig_img_resized)
    superimposed_img = np.uint8(0.6 * np_img + 0.4 * heatmap)

    # Plot & Save
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np_img)
    axes[0].set_title("Original X-Ray")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Attention Heatmap")
    axes[1].axis("off")

    axes[2].imshow(superimposed_img)
    diagnosis = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    axes[2].set_title(f"Grad-CAM: {diagnosis} ({prob*100:.1f}%)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_output_path, dpi=300)
    plt.close()

    print(f"Grad-CAM successfully generated and saved to: {save_output_path}")

if __name__ == "__main__":
    test_pneumonia_dir = "./data/test/PNEUMONIA"
    pneumonia_images = [f for f in os.listdir(test_pneumonia_dir) if not f.startswith('.')]
    
    if pneumonia_images:
        sample_img = os.path.join(test_pneumonia_dir, pneumonia_images[0])
        generate_gradcam(sample_img)
    else:
        print("Test folder me Pneumonia images nahi mili.")