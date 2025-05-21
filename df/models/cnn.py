from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch




class YourCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input: (3, 32, 32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)     # -> (32, 32, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)                              # -> (32, 16, 16)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)    # -> (64, 16, 16)
        self.bn2 = nn.BatchNorm2d(64)                               # -> (64, 8, 8)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)   # -> (128, 8, 8)
        self.bn3 = nn.BatchNorm2d(128)                              # -> (128, 4, 4)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)  # CIFAR-10 has 10 classes
        

    def forward(self, x):
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


    