import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from custom_module import CustomVGG19  # Replace with the correct import statement

# Set the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preprocessing and DataLoader setup for validation dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG19-compatible image size
    transforms.ToTensor(),
])

# Load the validation dataset
validation_dataset = datasets.ImageFolder(root='verification_dataset', transform=transform)

# DataLoader setup
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=4)

# Initialize the model
model = CustomVGG19(num_classes=2).to(device)
model.load_state_dict(torch.load('custom_vgg19_model.pth'))
model.eval()

# Loss function for validation
criterion = nn.CrossEntropyLoss()

# Validation loop
total_correct = 0
total_samples = 0

with torch.no_grad():
    for inputs, targets in validation_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Calculate accuracy
        _, predicted_class = torch.max(outputs, 1)
        total_correct += (predicted_class == targets).sum().item()
        total_samples += targets.size(0)

# Calculate accuracy
accuracy = total_correct / total_samples
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
