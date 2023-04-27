# Author: Kauthar Orrie
# Student no.: ORRKAU001
# Implementation of a LeNet5 - A Convolutional Neural Network (CNN) using PyTorch
# 2023
# CIFAR10 dataset
# Goal: 65% accuracy for the MLP

# Requried imports for PyTorch
import torch  # Main Package
import torchvision  # Package for Vision Related ML
import torchvision.transforms as transforms  # Subpackage that contains image transforms

import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions

import torch.optim as optim # Optimizers


# Create the transform sequence
# A container that contains a sequence of transform sequence
# transforms the image to tensors
# data needs to be transfomed to a tensor
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to Tensor
    # used 3 values for the CIFAR10 dataset because it is RGB
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
])
#normalise is to change the value of pixel to 1 or -1
# for assignment we can chnage the colours or normalise them with different mean and deviatoion

# Load CIFAR-10 dataset
# download the dataset using torch vision

# Download the Train set
# set train to TRUE
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# Download the Test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# the size of the batch you want to train on each time
BATCH_SIZE = 500 # global variable

# Send data to the data loaders
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# outside of CNN class
# send all the parameters or output to the right place, this code checks which device you have
# Identify device
device = ("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Define the training and testing functions
def train(net, train_loader, criterion, optimizer, device):
    net.train()  # Set model to training mode.
    running_loss = 0.0  # To calculate loss across the batches
    for data in train_loader:
        inputs, labels = data  # Get input and labels for batch
        inputs, labels = inputs.to(device), labels.to(device)  # Send to device
        optimizer.zero_grad()  # Zero out the gradients of the ntwork i.e. reset
        outputs = net(inputs)  # Get predictions
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Propagate loss backwards
        optimizer.step()  # Update weights
        running_loss += loss.item()  # Update loss
    return running_loss / len(train_loader)

def test(net, test_loader, device):
    net.eval()  # We are in evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Don't accumulate gradients
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # Send to device
            outputs = net(inputs)  # Get predictions
            _, predicted = torch.max(outputs.data, 1)  # Get max value
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # How many are correct?
    return correct / total

# Included in the CNN architecture:
# 1. A convolutional layer with a filter size of (5x5) and 6 output channels. 
# 2. A MaxPool subsampling layer with filter size of (2x2) and a stride of 2.
# 3. Another convolutional layer with a filter size of (5x5), 6 input channels and 16 output
# channels.
# 4. A MaxPool subsampling layer with filter size of (2x2) and a stride of 2.
# 5. A fully-connected layer with shape (400, 120).
# 6. A fully-connected layer with shape (120, 84).
# 7. A fully-connected layer with shape (84, 10)

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # for assign, we need to set first param to 3 (3 colour channels - RGB)
        # check documentation
        #padding - adds a black border around the image, no added padding in my architecture
        self.conv1 = nn.Conv2d(3, 6, 2, padding=0) # First Conv Layer
        self.pool1 = nn.MaxPool2d(2)  # For pooling
        self.conv2 = nn.Conv2d(6, 16, 3, padding=0) # Second Conv Layer
        self.pool2 = nn.MaxPool2d(2)  # For pooling
        self.bn1 = nn.BatchNorm2d(6) # used to increase performance of the CNN

    
        self.flatten = nn.Flatten() # For flattening the 2D image
        self.fc1 = nn.Linear(576, 400)  # First FC HL
        self.fc2 = nn.Linear(400, 120)  # First FC HL
        self.fc3= nn.Linear(120, 84) # Hidden
        self.drop1 = nn.Dropout(p=0.2) #find a lower p 
        self.fc4= nn.Linear(84, 10) # Output layer
  #forward pass
    def forward(self, x):
      # Batch x of shape (B, C, W, H)
      x = F.relu(self.conv1(x)) # Shape: (B, 5, 16, 16)
      x = self.pool1(x)  # Shape: (B, 5, 16, 16)
      x = self.bn1(x)
      x = F.relu(self.conv2(x)) # Shape: (B, 4, 12, 12)
      x = self.pool2(x)  # Shape: (B, 9, 8, 8)
      x = self.flatten(x) # Shape: (B, 576)
      x = F.relu(self.fc1(x))  # Shape (B, 400)
      x = self.drop1(x) # apply dropout
      x = F.relu(self.fc2(x))  # Shape (B, 120)
      x = F.relu(self.fc3(x))  # Shape (B, 84)
      # Output Layer
      x = self.fc4(x)  # Shape: (B, 10) #send to the output layer
      return x  

# Create the model and send its parameters to the appropriate device
cnn = CNN().to(device)

LEARNING_RATE = 1e-1
MOMENTUM = 0.9

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss() # Use this if not using softmax layer
optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# Apply a learning rate decay of 0.1 after every 6th epoch 
# Learning Rate Decay: the learing rate is reduced by 0.1
lr_decay = optim.lr_scheduler.StepLR(optimizer, 6, 0.1)

# Train the MLP for 15 epochs
for epoch in range(15):
    train_loss = train(cnn, train_loader, criterion, optimizer, device)
    test_acc = test(cnn, test_loader, device)
    lr_decay.step() # Apply the learning rate decay after every 6th epoch
    print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
