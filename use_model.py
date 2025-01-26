import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Define the CNN Model (must match the model used during training)
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Adjusted for CIFAR-10
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Load the saved model
def load_model(model_path):
    model = CNN(num_classes=10)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')  # Ensure 3-channel RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Predict the class of an image
def predict_image(model, image_path, class_names):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]


# Main function for the app
def main():
    model_path = 'cnn_model.pth'  # Path to the saved model
    if not os.path.exists(model_path):
        print("Model file not found. Please ensure 'cnn_model.pth' exists.")
        return

    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]  # CIFAR-10 class labels

    model = load_model(model_path)

    image_path = input("Enter the path to the image file: ")
    if not os.path.exists(image_path):
        print("Image file not found. Please enter a valid path.")
        return

    prediction = predict_image(model, image_path, class_names)
    print(f"The predicted class for the image is: {prediction}")


if __name__ == '__main__':
    main()
