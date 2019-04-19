"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs (number of batches? )
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

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        # self.weight_decay = 0.0001
        # self.weight_scale = 0.2
        # self.hidden_number = len(self.n_hidden)

        # Init linear modules and their activations
        self.linear_modules = []
        self.activation_modules = []

        # The first layer gets the input dimensions
        self.linear_modules.append(LinearModule(n_inputs, n_hidden[0]))
        self.activation_modules.append(ReLUModule())

        # number of neurons in all layers N-1 is specified  in n_hidden
        for i in range(len(self.n_hidden) - 1):
            self.linear_modules.append(LinearModule(n_hidden[i], n_hidden[i + 1]))
            self.activation_modules.append(ReLUModule())

        # The last linear layer will output n_classes
        self.linear_modules.append(LinearModule(n_hidden[-1], n_classes))
        # Last layer, softmax
        # self.softmax_module = SoftMaxModule()
        self.activation_modules.append(SoftMaxModule())
        self.softmax_probs = None

        # End loss for all batches or vector of losses with elements
        self.cross_entropy_module = CrossEntropyModule()
        # does not matter how you init
        self.cross_entropy_loss = 9999999999

        # how many layers in total: h_hidden  ( dont count entropy loss )
        self.n_modules = len(self.linear_modules)

        # To store previous derivatives and inputs (dout and x)
        self._douts = None
        self._input = None

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network (tensor, ravel the image fucking shit)
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        # we receive a pre-processed batch of [B x n_input_dims]
        for layer in range(len(self.linear_modules)):
            x = self.linear_modules[layer].forward(x)

            x = self.activation_modules[layer].forward(x)

        # self.cross_entropy_loss = self.cross_entropy_module.forward(x)
        out = x

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        # zzzup, do it in the back (order)

        for i in range(self.n_modules):
            dout = self.activation_modules[-(i + 1)].backward(dout)
            dout = self.linear_modules[-(i + 1)].backward(dout)

        return
