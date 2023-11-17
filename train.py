import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import models  # models 모듈 추가

# Define the custom VGG19 model
class CustomVGG19(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomVGG19, self).__init__()
        
        # Load pre-trained VGG19 model
        vgg19_model = models.vgg19(pretrained=True)
        
        # Extract feature layers (excluding fully connected layers)
        self.features = vgg19_model.features
        
        # Add a custom fully connected layer for classification
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Set the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preprocessing and DataLoader setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG19-compatible image size
    transforms.ToTensor(),
])

# Load the dataset
train_dataset = datasets.ImageFolder(root='dataset', transform=transform)

# DataLoader setup
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# Initialize the model
model = CustomVGG19(num_classes=2).to(device)

# Loss function and optimizer setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'custom_vgg19_model.pth')
