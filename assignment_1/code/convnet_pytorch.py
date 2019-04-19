"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_batchnorm import *


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels, n_classes):
        """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """
        super().__init__()

        # stores the input and output sizes
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Padding
        padding_avg_pooling = 0
        padding_all = 1

        # Kernel size of the conv layers
        kernel_size_avg_pooling = 1
        kernel_size = 3

        out_filters_dict = {"conv1": 64, "maxpool1": 64, "conv2": 128, "maxpool2": 128, "conv3": 256, "maxpool3": 256,
                            "conv4": 512, "maxpool4": 512, "conv5": 512, "maxpool5": 512, "avgpool": 512}

        strides_dict = {"conv": 1, "maxpool": 2, "avgpool": 1}

        self.conv1 = torch.nn.Conv2d(in_channels=n_channels,
                                     out_channels=out_filters_dict["conv1"],
                                     kernel_size=kernel_size,
                                     stride=strides_dict["conv"],
                                     padding=padding_all)

        # self.norm1 = torch.nn.BatchNorm2d(out_filters_dict["conv1"])
        self.norm1 = CustomBatchNormAutograd(out_filters_dict["conv1"])

        self.pool1 = torch.nn.MaxPool2d(kernel_size=kernel_size,
                                        stride=strides_dict["maxpool"],
                                        padding=padding_all)

        self.conv2 = torch.nn.Conv2d(in_channels=out_filters_dict["conv1"],
                                     out_channels=out_filters_dict["conv2"],
                                     kernel_size=kernel_size,
                                     stride=strides_dict["conv"],
                                     padding=padding_all)

        self.norm2 = torch.nn.BatchNorm2d(out_filters_dict["conv2"])

        self.pool2 = torch.nn.MaxPool2d(kernel_size=kernel_size,
                                        stride=strides_dict["maxpool"],
                                        padding=padding_all)

        self.conv3_a = torch.nn.Conv2d(in_channels=out_filters_dict["conv2"],
                                       out_channels=out_filters_dict["conv3"],
                                       kernel_size=kernel_size,
                                       stride=strides_dict["conv"],
                                       padding=padding_all)

        self.norm3 = torch.nn.BatchNorm2d(out_filters_dict["conv3"])

        self.conv3_b = torch.nn.Conv2d(in_channels=out_filters_dict["conv3"],
                                       out_channels=out_filters_dict["conv3"],
                                       kernel_size=kernel_size,
                                       stride=strides_dict["conv"],
                                       padding=padding_all)

        self.pool3 = torch.nn.MaxPool2d(kernel_size=kernel_size,
                                        stride=strides_dict["maxpool"],
                                        padding=padding_all)

        self.conv4_a = torch.nn.Conv2d(in_channels=out_filters_dict["conv3"],
                                       out_channels=out_filters_dict["conv4"],
                                       kernel_size=kernel_size,
                                       stride=strides_dict["conv"],
                                       padding=padding_all)

        self.norm4 = torch.nn.BatchNorm2d(out_filters_dict["conv4"])

        self.conv4_b = torch.nn.Conv2d(in_channels=out_filters_dict["conv4"],
                                       out_channels=out_filters_dict["conv4"],
                                       kernel_size=kernel_size,
                                       stride=strides_dict["conv"],
                                       padding=padding_all)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=kernel_size,
                                        stride=strides_dict["maxpool"],
                                        padding=padding_all)

        self.conv5_a = torch.nn.Conv2d(in_channels=out_filters_dict["conv4"],
                                       out_channels=out_filters_dict["conv5"],
                                       kernel_size=kernel_size,
                                       stride=strides_dict["conv"],
                                       padding=padding_all)

        self.norm5 = torch.nn.BatchNorm2d(out_filters_dict["conv5"])

        self.conv5_b = torch.nn.Conv2d(in_channels=out_filters_dict["conv5"],
                                       out_channels=out_filters_dict["conv5"],
                                       kernel_size=kernel_size,
                                       stride=strides_dict["conv"],
                                       padding=padding_all)

        self.pool5 = torch.nn.MaxPool2d(kernel_size=kernel_size,
                                        stride=strides_dict["maxpool"],
                                        padding=padding_all)

        self.pool6 = torch.nn.AvgPool2d(kernel_size=kernel_size_avg_pooling,
                                        stride=strides_dict["avgpool"],
                                        padding=padding_avg_pooling)

        # Finally fully connected
        self.fc = torch.nn.Linear(out_filters_dict["conv5"], n_classes)

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed
        through several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        # Attach non-linearities

        x = F.relu(self.norm1(self.conv1(x)))

        x = self.pool1(x)

        x = F.relu(self.norm2(self.conv2(x)))

        x = self.pool2(x)

        x = F.relu(self.norm3(self.conv3_a(x)))
        x = F.relu(self.norm3(self.conv3_b(x)))

        x = self.pool3(x)

        x = F.relu(self.norm4(self.conv4_a(x)))
        x = F.relu(self.norm4(self.conv4_b(x)))

        x = self.pool4(x)

        x = F.relu(self.norm5(self.conv5_a(x)))
        x = F.relu(self.norm5(self.conv5_b(x)))

        x = self.pool5(x)

        x = self.pool6(x)

        # no forget fully connected
        out = self.fc(x.view(x.shape[0], -1))

        return out
