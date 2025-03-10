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
from CNNangle import ClimbingGradeCNN

# Load the dataset
df = pd.read_csv("csv_data/grades40_noplus.csv")

# Custom Dataset Class
class ClimbingGradeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]  # Assuming the image path is in the first column
        label = self.df.iloc[idx, 1]     # Assuming the grade is in the second column
        degree = self.df.iloc[idx, 2]    # Assuming the degree is in the third column

        # Load the image
        image = Image.open('cropped40/'+img_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        if label == None:
            print(img_path)
        # Convert the label to an integer and ensure it's a torch.LongTensor
        label = int(label)  # Ensure the label is an integer
        degree = int(degree)
        degree = torch.tensor(float(degree), dtype=torch.float32)


        return image, label, degree

# Define image transformations (resize and to tensor)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Create the dataset
dataset = ClimbingGradeDataset(df, transform=transform)

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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Move model to MPS
model = ClimbingGradeCNN().to(device)

import torch.optim as optim
import torch.nn as nn


# Example of class frequencies
#class_counts = [76,1,116,1,72,1,103,1,118,1,196,1,417,216,380,239,438,256,607,457,431,181,205,115,41,7,24,1,42]
#total_samples = sum(class_counts)

# Calculate class weights (inverse frequency)
#weights = [total_samples / (len(class_counts) * count) for count in class_counts]
#weights = torch.tensor(weights, dtype=torch.float)

# Define the loss function with weights
criterion = nn.CrossEntropyLoss()


# Example model
model = ClimbingGradeCNN()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
training_loss =[]
validation_loss = []
import torch

# Set the device to MPS (Apple Silicon GPU) if available, otherwise use CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Move model to MPS
model = ClimbingGradeCNN().to(device)

# Training loop
num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, labels, degrees) in enumerate(train_loader):  
        optimizer.zero_grad()

        # Move inputs, labels, and degrees to the same device as the model
        inputs = inputs.to(device)
        labels = labels.to(device)
        degrees = degrees.to(device)

        # Forward pass with degree input
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)
        accuracy = (outputs.argmax(dim=1) == labels).float().mean()
        print(f"Training Accuracy: {accuracy}")

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0.0
    outputs_list, labels_list = [], []

    with torch.no_grad():
        for batch_idx, (inputs, labels, degrees) in enumerate(val_loader):
            # Move inputs, labels, and degrees to MPS
            inputs = inputs.to(device)
            labels = labels.to(device)
            degrees = degrees.to(device)

            # Forward pass with degree input
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            val_accuracy = (outputs.argmax(dim=1) == labels).float().mean()
            print(f"Validation Accuracy: {val_accuracy}")

            outputs_list.append(outputs.cpu())  # Move outputs back to CPU for evaluation
            labels_list.append(labels.cpu())    # Move labels back to CPU for evaluation

    # Print loss after each epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}")
    training_loss.append(running_loss / len(train_loader))
    validation_loss.append(val_loss / len(val_loader))

#import matplotlib.pyplot as plt
#plt.plot(training_loss, label='Training Loss')
#plt.plot(validation_loss, label='Validation Loss')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()

#make a confusion matrix
from sklearn.metrics import confusion_matrix
outputs = torch.cat(outputs_list)
labels = torch.cat(labels_list)
conf_matrix = confusion_matrix(labels, outputs.argmax(dim=1))
#show confusion matrix as a heatmap
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.show()

