import os

base_dir = "./data/train"
classes = ["NORMAL", "PNEUMONIA"]

print("--- Checking Training Dataset Distribution ---")
total_images = 0

for cls in classes:
    class_folder = os.path.join(base_dir, cls)
    if os.path.exists(class_folder):
        files = [f for f in os.listdir(class_folder) if not f.startswith('.')]
        count = len(files)
        total_images += count
        print(f"Class '{cls}': {count} images")
    else:
        print(f"Folder nahi mila: {class_folder}")

print(f"Total Images: {total_images}")