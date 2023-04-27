import torch  # Main Package
import torchvision  # Package for Vision Related ML
import torchvision.transforms as transforms  # Subpackage that contains image transforms

import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions

import torch.optim as optim # Optimizers



# Create the transform sequence
# container that contains a sequence of transform sequence
# transforms the image to tensors
# data needs to be transfomed to a tensor
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to Tensor
    # used 3 values for the CIFAR10 dataset because it is RGB
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
])
#normalise is to change the value of pixel to 1 or -1
#for assignment we can chnage the colours or normalise them with different mean and deviatoion

# Load CIFAR-10 dataset
# download the dataset using torch vision

# Train
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# Test
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# the batch size you want to train on at a time
# Send data to the data loaders
BATCH_SIZE = 2
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# outside of MLP class
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

# 1. A convolutional layer with a filter size of (5x5) and 6 output channels. You will need
# to determine the number of input channels based on shape of your data.
# 2. A MaxPool subsampling layer with filter size of (2x2) and a stride of 2.
# 3. Another convolutional layer with a filter size of (5x5), 6 input channels and 16 output
# channels.
# 4. A MaxPool subsampling layer with filter size of (2x2) and a stride of 2.
# 5. A fully-connected layer with shape (400, 120).
# 6. A fully-connected layer with shape (120, 84).
# 7. A fully-connected layer with shape (84, 10
# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # for assign, we need to set first param to 3 (3 colour channels)
        # check documentation
        #padding - adds a black border around the image
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1) # First Conv Layer
        self.pool1 = nn.MaxPool2d(2)  # For pooling
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1) # First Conv Layer
        self.pool2 = nn.MaxPool2d(2)  # For pooling
        self.flatten = nn.Flatten() # For flattening the 2D image
        self.fc1 = nn.Linear(400, 120)  # First FC HL
        self.fc2= nn.Linear(120, 84) # Hidden
        self.fc3= nn.Linear(84, 10) # Output layer
  #forward pass
    def forward(self, x):
      # Batch x of shape (B, C, W, H)
      x = F.relu(self.conv1(x)) # Shape: (B, 5, 28, 28)
      x = self.pool1(x)  # Shape: (B, 5, 14, 14)
      x = F.relu(self.conv2(x)) # Shape: (B, 5, 28, 28)
      x = self.pool2(x)  # Shape: (B, 5, 14, 14)
      x = self.flatten(x) # Shape: (B, 980)
      x = F.relu(self.fc1(x))  # Shape (B, 256)
      x = F.relu(self.fc2(x))  # Shape (B, 256)
      x = self.fc3(x)  # Shape: (B, 10) #send to the output layer
      return x  

# cnn = CNN().to(device)

# LEARNING_RATE = 1e-1
# MOMENTUM = 0.9

# # Define the loss function, optimizer, and learning rate scheduler
# criterion = nn.CrossEntropyLoss() # Use this if not using softmax layer
# optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# # Train the MLP for 5 epochs
# for epoch in range(5):
#     train_loss = train(cnn, train_loader, criterion, optimizer, device)
#     test_acc = test(cnn, test_loader, device)
#     print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")

#     #overfitting is detected when the train loss is decreasing but the accuracy is becoming worse
#     #apply the softmax to get the probabliity