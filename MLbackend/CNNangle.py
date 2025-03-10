import torch
import torch.nn as nn
import torch.nn.functional as F

class ClimbingGradeCNN(nn.Module):
    def __init__(self):
        super(ClimbingGradeCNN, self).__init__()
        
        # Convolutional layers with Batch Normalization
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
        
        # Fully connected layers for image features
        self.fc1 = nn.Linear(16400, 528)  # Assuming input image size is 700x700
        self.fc2 = nn.Linear(528, 256)  # +16 from angle MLP
        self.fc3 = nn.Linear(256, 29)  # Number of classes
        
        # Small MLP for the angle input
        self.angle_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # Dropout layers
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x,angle):
        # Convolutional layers with Batch Normalization, ReLU, and MaxPooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn6(self.conv6(x)))

        # Flatten CNN output
        x = x.view(x.size(0), -1)


        # Process the angle input
        angle_input = torch.tensor(angle, dtype=torch.float32) 
        angle_input = angle_input.view(-1, 1)  # Ensure correct shape
        angle_features = self.angle_fc(angle_input)

        # Concatenate CNN features with angle features
        x = torch.cat((x, angle_features), dim=1)

        # Fully connected layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)

        return x
