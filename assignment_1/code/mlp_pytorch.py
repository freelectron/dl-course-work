"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        # # ===============================  Approach 1  ===========================================
        #
        # # get access to pytorch.nn
        # super().__init__()
        #
        # # dims of the inputs
        # self.n_inputs = n_inputs
        # self.n_hidden = n_hidden
        # self.n_classes = n_classes
        #
        # # define the network architecture
        # self.linear_modules = list()
        # self.activation_modules = list()
        #
        # # layer 1 to map from input to the first hidden
        # self.linear_modules.append(nn.Linear(self.n_inputs, self.n_hidden[0]))
        # # self.activation_modules.append(nn.ReLU())
        #
        # for i in range(len(n_hidden) - 1):
        #     self.linear_modules.append(nn.Linear(self.n_hidden[i], self.n_hidden[i]))
        #     # self.activation_modules(nn.ReLU())
        #
        # # the last one
        # self.linear_modules.append(nn.Linear(self.n_hidden[-1], self.n_classes))
        # # self.activation_modules.append(nn.Softmax())
        #
        # self.modules_list = nn.ModuleList(self.linear_modules)
        #
        # self.model_params_tensors = [list((next(self.linear_modules[i].parameters()))) for i in
        #                              range(len(self.linear_modules))]
        #
        # # make pytorch see the modules
        # self.param_list = nn.ParameterList([nn.Parameter(torch.tensor(next(self.modules_list[0].parameters()))),
        #                                     nn.Parameter(torch.tensor(next(self.modules_list[1].parameters())))])

        # ===============================  Approach 1.2, sequantial  ===========================================

        # get access to pytorch.nn
        super().__init__()

        # dims of the inputs
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        # define the network architecture
        self.modules = list()
        self.modules = list()

        # layer 1 to map from input to the first hidden
        self.modules.append(nn.Linear(self.n_inputs, self.n_hidden[0]))
        self.modules.append(nn.ReLU())

        for i in range(len(n_hidden) - 1):
            self.modules.append(nn.Linear(self.n_hidden[i], self.n_hidden[i+1]))
            self.modules.append(nn.ReLU())

        # the last one
        self.modules.append(nn.Linear(self.n_hidden[-1], self.n_classes))
        # self.modules.append(nn.Softmax())

        print(self.modules)

        # Create sequential model :(
        self.model = nn.Sequential(*self.modules)


    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """
        # ===============================  Approach 1  ===========================================
        # for i in range(len(self.linear_modules)):
        #     if i < len(self.linear_modules) - 1:
        #         x = F.relu(self.linear_modules[i](x))
        #
        # x = self.linear_modules[i](x)
        # out = x

        # ===============================  Approach 2  ===========================================
        out = self.model(x)

        return out
