import torch
import torch.nn as nn

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
        self.conv3 = nn.Conv2d(input_channels, output_channels*self.expansion, kernel_size=1, stride=1, padding=0)
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



