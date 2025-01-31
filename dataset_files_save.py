import os
from torchvision import datasets

# Define paths
save_dir = "cifar10_images"
os.makedirs(save_dir, exist_ok=True)

# Download the CIFAR-10 dataset
dataset = datasets.CIFAR10(root="./data", train=True, download=True)

# Save images locally
for idx, (image, label) in enumerate(dataset):
    # Create a directory for each label
    label_dir = os.path.join(save_dir, str(label))
    os.makedirs(label_dir, exist_ok=True)
    
    # Save the image directly (already a PIL image)
    image.save(os.path.join(label_dir, f"{idx}.png"))

    # Optional: Limit saved images for testing
    if idx >= 99:  # Save the first 100 images only (optional)
        break

print(f"Saved CIFAR-10 images to '{save_dir}'")
