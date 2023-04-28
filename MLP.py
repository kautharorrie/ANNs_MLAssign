# Author: Kauthar Orrie
# Student no.: ORRKAU001
# Implementation of a Multilayer perceptron using PyTorch
# 2023
# CIFAR10 dataset
# Goal: 58% accuracy for the MLP

# requried imports for PyTorch
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
#for assignment we can chnage the colours or normalise them with different mean and deviatoion

# Load CIFAR-10 dataset
# download the dataset using torch vision

# Download the Train set
# set train to TRUE
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# Download the Test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


# Define the training and testing functions
def train(net, train_loader, criterion, optimizer, device):
    net.train()  # Set model to training mode.
    running_loss = 0.0  # To calculate loss across the batches
    for data in train_loader:
        inputs, labels = data  # Get input and labels for batch
        inputs, labels = inputs.to(device), labels.to(device)  # Send to device
        #everytime you want to start a new batch set gradient to zero
        optimizer.zero_grad()  # Zero out the gradients of the ntwork i.e. reset
        #next for lines are very important
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

# the size of the batch you want to train on each time
BATCH_SIZE = 500 # global variable

# Send data to the data loaders
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# outside of MLP class
# send all the parameters or output to the right place, this code checks which device you have
# Identify device
device = ("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Define the Multi-layer perceptron architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # need to flatten the images before training
        self.flatten = nn.Flatten() # For flattening the 2D image

        #input layer + 2 hidden layers + output layer
        # input layer (fc1) + HL 1 (fc2) + HL 2 (fc3) + output layer (fc4)
        # in brackets: (input layer size, output layer size)
        # Input is image with shape (32*32) with an input channel of 3 (RGB) 
        self.fc1 = nn.Linear(32*32*3, 2048)  # output size of 2048 is 2/3 of the input layer (most optimal param for this dataset to achieve 58% accuracy)
        self.fc2 = nn.Linear(2048, 1024)  # First HL hidden layer
        self.fc3= nn.Linear(1024, 512) # Second HL hidden layer
        # self.fc4= nn.Linear(512, 256) # Third HL hidden layer
        self.fc5= nn.Linear(512, 10) # Output
        self.drop1 = nn.Dropout(p=0.1) #find a lower p 
        self.output = nn.LogSoftmax(dim=1)

    # override the forward method and implement the logic of MLP
    def forward(self, x):
      # Batch x of shape (B, C, W, H)
      x = self.flatten(x) # Batch now has shape (B, C*W*H) # flatten the 2D shape to 1D array
      x = F.relu(self.fc1(x))  # Input Layer, apply the ReLu activation function on the nodes
      x = self.drop1(x) # apply droput at the input layer to increase accuracy of MLP
      x = F.relu(self.fc2(x))  # First Hidden Layer, apply the ReLu activation function on the nodes
      x = F.relu(self.fc3(x)) # Second Hidden Layer, apply the ReLu activation function on the nodes
    #   x = F.relu(self.fc4(x)) # Third Hidden Layer, apply the ReLu activation function on the nodes
      x = self.fc5(x)  # Output Layer
      x = self.output(x)  # For multi-class classification
      return x  # Has shape (B, 10)

# Create the model and send its parameters to the appropriate device
mlp = MLP().to(device) #multi layer perceptron class being called

# you can change the learning rate to get better accuracy
LEARNING_RATE = 1e-1
MOMENTUM = 0.9

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.NLLLoss()
optimizer = optim.SGD(mlp.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# Apply a learning rate decay of 0.1 after every 6th epoch 
# Learning Rate Decay: the learing rate is reduced by 0.1
lr_decay = optim.lr_scheduler.StepLR(optimizer, 6, 0.1) 

# Train the MLP for 15 epochs
for epoch in range(15):
    train_loss = train(mlp, train_loader, criterion, optimizer, device)
    test_acc = test(mlp, test_loader, device)
    lr_decay.step() # Apply the learning rate decay after every 6th epoch
    print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")


