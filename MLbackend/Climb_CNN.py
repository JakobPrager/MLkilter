# climbinggrade CNN
import torch.nn as nn
import torch.nn.functional as F

class ClimbingGradeCNN(nn.Module):
    def __init__(self):
        super(ClimbingGradeCNN, self).__init__()
        
        # Convolutional layers with Batch Normalization and Dropout
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)
        
        # Fully connected layers with dropout
        
        #self.fc1 = nn.Linear(2048 * 4 * 4, 1024)
        self.fc1 = nn.Linear(1024 * 4 * 4, 512)  # Assuming input image size is 700x700
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # For 10 classes (change based on your number of grades)

        # Dropout layers
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Convolutional layers with Batch Normalization, ReLU, and MaxPooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # MaxPool with 2x2 kernel
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn6(self.conv6(x)))
        #x = F.max_pool2d(x, 2)

        

        # Flatten the output for fully connected layers
        #print(x.shape) 
        x = x.view(x.size(0), -1)


        # Fully connected layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)

        return x
