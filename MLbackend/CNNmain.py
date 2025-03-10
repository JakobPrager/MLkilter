#CNN approach for classifying the kilter dataset

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from CustomLoss import ordinal_cross_entropy, grade_distance_loss, combined_loss
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from Climb_CNN import ClimbingGradeCNN

# Load the dataset
df = pd.read_csv("csv_data/grades40_noplus.csv")

from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

# Define your augmentation and preprocessing pipeline
augmentation_transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # Random crop and resize to 128x128
    transforms.RandomRotation(5),                  # Random horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color variations
    transforms.ToTensor(),                               # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])

# Custom Dataset Class with Data Augmentation
class ClimbingGradeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        # Use the provided transform or default to our augmentation transform
        self.transform = transform if transform is not None else augmentation_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]  # Assuming the image path is in the first column
        label = self.df.iloc[idx, 1]     # Assuming the grade is in the second column
        degree = self.df.iloc[idx, 2]    # Assuming the degree is in the third column

        # Load the image and convert it to RGB
        image = Image.open('rectangle40/' + img_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        if label is None:
            print(img_path)
        # Convert the label to an integer and ensure it's a torch.LongTensor
        label = int(label)
        return image, label


"""# Define image transformations (resize and to tensor)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])"""

# Create the dataset
dataset = ClimbingGradeDataset(df, transform=augmentation_transform)

# Calculate the sizes for train, validation, and test splits (e.g., 80%, 10%, 10%)
total_size = len(dataset)
train_size = int(0.8 * total_size)  # 80% for training
val_size = int(0.1 * total_size)    # 10% for validation
test_size = total_size - train_size - val_size  # Remaining 10% for testing

# Randomly split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each split
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# Example instantiation
model = ClimbingGradeCNN()
print(model)

import torch.optim as optim
import torch.nn as nn

# Set the device to MPS (Apple Silicon GPU) if available, otherwise use CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Move model to MPS
model = ClimbingGradeCNN().to(device)


# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
training_loss =[]
validation_loss = []
# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Training phase
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(inputs)
        outputs = outputs.squeeze(1) 
        labels = labels.to(torch.float32) 
        loss = criterion(outputs, labels)   
        # Round the output to the nearest integer
        outputs_rounded = torch.round(outputs)
        # Compute accuracy by comparing rounded outputs to labels
        accuracy = (outputs_rounded.squeeze() == labels).float().mean()
        print(f"Training Accuracy: {accuracy}")
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation phase (at the end of each epoch)
    model.eval()
    val_loss = 0.0
    outputs_list = []
    labels_list = []
    with torch.no_grad():  # No gradient computation for validation
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            outputs = model(inputs)
            outputs = outputs.squeeze(1) 
            labels = labels.to(torch.float32) 
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # Round the output to the nearest integer
            outputs_rounded = torch.round(outputs)
            # Compute accuracy by comparing rounded outputs to labels
            val_accuracy = (outputs_rounded.squeeze() == labels).float().mean()

            print(f"Validation Accuracy: {val_accuracy}")
            outputs_list.append(outputs.cpu())  # Move outputs back to CPU for evaluation
            labels_list.append(labels.cpu())    # Move labels back to CPU for evaluation

    # Print loss after each epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}")
    training_loss.append(running_loss / len(train_loader))
    validation_loss.append(val_loss / len(val_loader))

#save model 
torch.save(model.state_dict(), 'climbing_grade_cnn_aug.pth')


import matplotlib.pyplot as plt
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#make a confusion matrix
from sklearn.metrics import confusion_matrix
outputs = torch.cat(outputs_list)
labels = torch.cat(labels_list)
conf_matrix = confusion_matrix(labels, torch.round(outputs))
#show confusion matrix as a heatmap
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.show()

