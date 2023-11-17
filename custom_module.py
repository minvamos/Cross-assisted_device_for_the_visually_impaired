import torch.nn as nn
import torchvision.models as models

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
