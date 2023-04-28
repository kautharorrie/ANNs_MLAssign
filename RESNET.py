# Requried imports for PyTorch
import torch  # Main Package
import torchvision  # Package for Vision Related ML
import torchvision.transforms as transforms  # Subpackage that contains image transforms

import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions

import torch.optim as optim # Optimizers

# better to create a residual block because it will be used reused multiple times
# this is not the ResNet architecture but rather the residual block that will be implemented every
class ResBlock(nn.Module):
    # sample: a conv layer, that might have changed if the input/output layers is changed
    def __init__(self, input_channels, output_channels, sample=None, stride=1):
        super(ResBlock, self).__init__()
        self.expansion = 4 # the output channel will be multiplied by 4 at every  third conv layer
        # first conv layer
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(output_channels)
        # second conv layer
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        # at this point we multiply the output channels by 4
        self.conv3 = nn.Conv2d(output_channels, output_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(output_channels*self.expansion)
        self.relu = nn.ReLU()
        self.sample = sample

    def forward(self, x):
        identity = x

        # apply all the defined layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) # apply relu activation function between the conv layers
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        # check if the shape needs to be changed
        # if the sample is not none, we will have to run it through x
        if self.sample is not None:
            identity = self.sample(identity)

        x += identity
        x = self.relu(x)
        return x
    
# defining the ResNet architecture
class RESNET(nn.Module):
    # resblock - the residual block thats passed in
    # num_of_layers - tells us how much we want to reuse the residual block at each RESNET layer (can be a list)
    # imgchannels - number of channels - RGB - 3 input channels
    def __init__(self, resblock, num_of_layers, imgchannels, num_of_classes):
        super(RESNET, self).__init__()
        # define layers here, no implementation yet
        self.in_chan = 64
        self.conv1 = nn.Conv2d(imgchannels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #RESNET LAYERS
        self.layer1 = self.create_layer(resblock, num_of_layers[0], out_channels=64, stride=1 )
        self.layer2 = self.create_layer(resblock, num_of_layers[1], out_channels=128, stride=2 )
        self.layer3 = self.create_layer(resblock, num_of_layers[2], out_channels=256, stride=2 )
        self.layer4 = self.create_layer(resblock, num_of_layers[3], out_channels=512, stride=2 ) 

        self.averagepool = nn.AdaptiveAvgPool2d((1,1)) # to make sure its of the correct shape
        self.fc = nn.Linear(512*4, num_of_classes)


    def forward(self, x):
        # conv layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # implement the ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # apply average pool to make sure it is the correct shape
        x = self.averagepool(x)
        # apply reshape so that the shape can be passed to the fully connected layer
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x) # fully connected layer
        return x


    # to make it easier to define the ResNet layers, the create_layer function will be created
    # this function will create the layers
    # resblock - residual block, num_of_blocks - number of residual blocks needed, ou
    def create_layer(self, resblock, num_of_blocks, out_channels, stride):
        identity = None
        layers = []

        # if the stride is 2, change the shape of the conv layer
        if stride != 1 or self.in_chan != out_channels*4:
            identity = nn.Sequential(nn.Conv2d(self.in_chan, out_channels*4, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels*4))

        layers.append(resblock(self.in_chan, out_channels, identity, stride))
        self.in_chan = out_channels*4

        for i in range(num_of_blocks-1):
            layers.append(resblock(self.in_chan, out_channels))
        
        return nn.Sequential(*layers)



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


# Create the model and send its parameters to the appropriate device
resnet = RESNET(ResBlock, [3,4,6,3], 3, 1000).to(device)

LEARNING_RATE = 1e-1
MOMENTUM = 0.9

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss() # Use this if not using softmax layer
optimizer = optim.SGD(resnet.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# Apply a learning rate decay of 0.1 after every 6th epoch 
# Learning Rate Decay: the learing rate is reduced by 0.1
lr_decay = optim.lr_scheduler.StepLR(optimizer, 6, 0.1)

# Train the MLP for 15 epochs
for epoch in range(15):
    train_loss = train(resnet, train_loader, criterion, optimizer, device)
    test_acc = test(resnet, test_loader, device)
    lr_decay.step() # Apply the learning rate decay after every 6th epoch
    print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
