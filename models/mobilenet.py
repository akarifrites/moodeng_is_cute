import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10

class MobileNetV3Audio(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV3Audio, self).__init__()
        self.model = models.mobilenet_v3_small(pretrained=True)
        self.model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)  # Change input to 1 channel
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[0].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)
    
# Initialize Model
model = MobileNetV3Audio(num_classes).to(device)    