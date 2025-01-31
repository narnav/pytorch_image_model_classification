import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
import os
from PIL import Image
import numpy as np

# Define the model architecture (a simple CNN model using pre-trained ResNet)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)  # Adjust final layer for our classes

    def forward(self, x):
        return self.model(x)

# Create a model instance
model = SimpleCNN(num_classes=3)

# Define the device (use GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to match model input
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
])

def load_model(model_path='model.pth'):
    """Load a saved model"""
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set model to evaluation mode
        print(f"Model loaded from {model_path}")
    else:
        print(f"No model found at {model_path}. Please train first.")
    return model

def save_model(model, model_path='model.pth'):
    """Save the trained model"""
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def predict_image(image_path):
    """Predict the class of the image"""
    model.eval()
    
    # Open and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Get the prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    # Read class labels from 'labels.txt'
    with open('labels.txt', 'r') as f:
        labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    
    predicted_class = labels[predicted.item()]
    
    return predicted_class

def train_model(train_data, num_epochs=5, learning_rate=0.001):
    """Train the model with the provided dataset"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    
    save_model(model)  # Save the model after training

def get_train_data(train_dir='uploads'):
    """Load image data from a directory"""
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    return train_dataset
