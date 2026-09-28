import os
import torch
from PIL import Image
from torchvision import transforms
from model import build_model

# 1. Device & Model Load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(pretrained=False).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# 2. Image Preprocessing Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_xray(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' nahi mili!")
        return

    # Image open & convert to RGB
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

    # Diagnosis Decision (threshold 0.5)
    diagnosis = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    confidence = prob if prob >= 0.5 else (1 - prob)

    print("\n--- DIAGNOSIS REPORT ---")
    print(f"Image Path:  {image_path}")
    print(f"Prediction:  {diagnosis}")
    print(f"Confidence:  {confidence * 100:.2f}%")
    print(f"Raw Prob:    {prob:.4f}")
    print("------------------------\n")

if __name__ == "__main__":
    # Test folder me se kisi ek image ka path check karein
    test_normal_folder = "./data/test/NORMAL"
    sample_images = [f for f in os.listdir(test_normal_folder) if not f.startswith('.')]
    
    if sample_images:
        sample_path = os.path.join(test_normal_folder, sample_images[0])
        predict_xray(sample_path)
    else:
        print("Test folder me images nahi mili.")