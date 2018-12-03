# TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # TODO: Define all the layers of this CNN, the only requirements are:
        # 1. This network takes in a square (same width and height), grayscale image as input
        # 2. It ends with a linear layer that represents the keypoints
        # it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # output size = (224 - 5) / 1 + 1 = 220, 220 x 220 x 32
        self.conv1 = nn.Conv2d(1, 32, 5)

        # maxpool layer
        # kernel_size=2, stride=2
        # output size = 220 / 2 = 110, 110 x 110 x 32
        self.pool = nn.MaxPool2d(2, 2)

        # 32 input image channels, 64 output channels, 5x5 square kernel,
        # output size = (110 - 5) / 1 + 1 = 106
        self.conv2 = nn.Conv2d(32, 64, 5)

        # another pooling
        # output size = 108 / 2 = 54

        # 64 input image channels, 128 output channels, 5x5 square kernel,
        # output size = (54 - 5) / 1 + 1 = 50
        self.conv3 = nn.Conv2d(64, 128, 5)

        # another pooling
        # output size = 50 / 2 = 25

        # 64 input image channels, 128 output channels, 5x5 square kernel,
        # output size = (25 - 5) / 1 + 1 = 21
        self.conv4 = nn.Conv2d(128, 256, 5)

        # another pooling
        # output size = 21 / 2 = 10

        #  input image channels, 256 output channels, 5x5 square kernel,
        # output size = (10 - 3) / 1 + 1 = 8
        self.conv5 = nn.Conv2d(256, 256, 3)

        # another pooling
        # output size = 8 / 2 = 4

        self.fc1 = nn.Linear(4 * 4 * 256, 2720)

        self.fc1_drop = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(2720, 680)

        self.fc2_drop = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(680, 136)

        # Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch
        # normalization) to avoid overfitting

    def forward(self, x):
        # TODO: Define the feedforward behavior of this model
        # x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc1_drop(x)
        x = self.fc2(x)
        x = self.fc2_drop(x)
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
