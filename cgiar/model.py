import torch.nn as nn
import torchvision.models as models


class Resnet50_V1(nn.Module):
    def __init__(self):
        super(Resnet50_V1, self).__init__()
        # Load a pretrained CNN (e.g., ResNet50)
        self.cnn = models.resnet50(weights=None)
        
        # Modify the final classification layer to output a single regression value
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # Forward pass through the CNN and regression layers
        x = self.cnn(x)
        return x
    
class Resnet50_V2(nn.Module):
    def __init__(self):
        super(Resnet50_V2, self).__init__()
        # Load a pretrained CNN (e.g., ResNet50)
        self.cnn = models.resnet50(weights=None)
        
        # Modify the final classification layer to output a single regression value
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 101)
        )

    def forward(self, x):
        # Forward pass through the CNN and regression layers
        x = self.cnn(x)
        return x
    
class Resnet50_V3(nn.Module):
    def __init__(self):
        super(Resnet50_V3, self).__init__()
        # Load a pretrained CNN (e.g., ResNet50)
        self.cnn = models.resnet50(weights=None)
        
        # Modify the final classification layer to output a single regression value
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        # Forward pass through the CNN and regression layers
        x = self.cnn(x)
        return x