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

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # need to flatten the images before training
        self.flatten = nn.Flatten() # For flattening the 2D image

        # (input layer no., output layer)
        self.fc1 = nn.Linear(32*32*3, 64)  # Input is image with shape (28x28)
        # 64, 32
        # 512, 256
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Input is image with shape (28x28)
        self.fc2 = nn.Linear(64, 32)  # First HL hidden layer
        self.fc3= nn.Linear(32, 16) # Second HL hidden layer
        self.fc4= nn.Linear(16, 10) # Third HL hidden layer
        self.drop1 = nn.Dropout(p=0.1) #find a lower p 
        self.output = nn.LogSoftmax(dim=1)

    # override the forward method and implement the logic of MLP
    def forward(self, x):
      # Batch x of shape (B, C, W, H)
      x = self.flatten(x) # Batch now has shape (B, C*W*H)
      x = F.relu(self.fc1(x))  # Input Layer
      x = self.drop1(x)
    #   x = F.relu(self.fc2(x))  # Second Hidden Layer
    #   x = F.relu(self.fc3(x))
      x = F.tanh(self.fc2(x))  # Second Hidden Layer
      x = F.tanh(self.fc3(x))
    # 
    # x = self.drop1(x)
      x = self.fc4(x)  # Output Layer
      x = self.output(x)  # For multi-class classification
      return x  # Has shape (B, 10)

# Create the model and send its parameters to the appropriate device
mlp = MLP().to(device) #multi layer perceptron

# you can change the learning rate to get better accuracy
LEARNING_RATE = 1e-4
MOMENTUM = 0.9

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.NLLLoss()

# criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

lr_decay = optim.lr_scheduler.StepLR(optimizer, 6, 0.1)
# Train the MLP for 5 epochs
for epoch in range(15):
    train_loss = train(mlp, train_loader, criterion, optimizer, device)
    test_acc = test(mlp, test_loader, device)
    lr_decay.step()
    print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")


