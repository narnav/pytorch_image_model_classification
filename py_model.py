import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

# Define your model (e.g., ResNet50)
class ResNet50Model(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet50Model, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # Replace the final fully connected layer for your own classification problem
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load the model
def load_model():
    model = ResNet50Model(num_classes=3)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Predict the image
def predict_image(model, image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    # Load class names
    with open('labels.txt', 'r') as file:
        class_names = [line.strip().split(' ')[1] for line in file.readlines()]

    return class_names[predicted_class.item()]
