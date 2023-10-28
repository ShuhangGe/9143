import torch
import torch.nn as nn

class MiniGoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MiniGoogLeNet, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu1 = nn.ReLU(inplace=True)

        # Mini Inception module
        self.conv2_1x1 = nn.Conv2d(96, 32, 1)
        self.conv2_3x3 = nn.Conv2d(96, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        # Max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Fully connected layer
        self.fc = nn.Linear(64*14*14, num_classes)

    def forward(self, x):
        # First Layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Mini Inception Module
        branch1 = self.conv2_1x1(x)
        branch2 = self.conv2_3x3(x)
        x = torch.cat([branch1, branch2], 1)
        x = self.bn2(x)
        x = self.relu2(x)

        # Max pooling
        x = self.maxpool(x)

        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc(x)

        return x
