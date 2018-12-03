import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    '''
    This model is created based on the first version of AlexNet
    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    '''

    def __init__(self):
        super(Net, self).__init__()

        # formula
        # conv layer
        # output_size = (input_size - kernel_size + 2 * padding) / stride + 1
        # padding = ((output_size - 1) * stride) + kernel_size - input_size) / 2
        # pooling layer
        # output_size = (input_size - kernel_size) / stride + 1
        # where input_size and output_size are the square image side length

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(16, 128, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(128, 192, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(6 * 6 * 128, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 136),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 6 * 6 * 128)
        x = self.classifier(x)
        return x