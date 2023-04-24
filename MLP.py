import torch  # Main Package
import torchvision  # Package for Vision Related ML
import torchvision.transforms as transforms  # Subpackage that contains image transforms

import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions

# Create the transform sequence
# container that contains a sequence of transform sequence
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

# Train
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# Test
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# the batch size you want to train on at a time
# Send data to the data loaders
BATCH_SIZE = 128
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # need to flatten the images before training
        self.flatten = nn.Flatten() # For flattening the 2D image

        # (input layer no., output layer)
        self.fc1 = nn.Linear(32*32, 512)  # Input is image with shape (28x28)
        self.fc2 = nn.Linear(512, 256)  # First HL hidden layer
        self.fc3= nn.Linear(256, 10) # Second HL hidden layer
        self.output = nn.LogSoftmax(dim=1)

    # override the forward method and implement the logic of MLP
    def forward(self, x):
      # Batch x of shape (B, C, W, H)
      x = self.flatten(x) # Batch now has shape (B, C*W*H)
      x = F.relu(self.fc1(x))  # First Hidden Layer
      x = F.relu(self.fc2(x))  # Second Hidden Layer
      x = self.fc3(x)  # Output Layer
      x = self.output(x)  # For multi-class classification
      return x  # Has shape (B, 10)

# outside of MLP class
# send all the parameters or output to the right place, this code checks which device you have
# Identify device
device = ("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Create the model and send its parameters to the appropriate device
mlp = MLP().to(device) #multi layer perceptron


