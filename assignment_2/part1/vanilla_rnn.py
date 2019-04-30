################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from argparse import Namespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import *


class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu', **kwargs):
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device
        self.activation = nn.Tanh()

        # network learnable params
        self.W_hx = nn.init.kaiming_normal_(nn.Parameter(torch.rand(input_dim, num_hidden)))
        self.W_hh = nn.init.kaiming_normal_(nn.Parameter(torch.rand(num_hidden, num_hidden)))
        self.W_ph = nn.init.kaiming_normal_(nn.Parameter(torch.rand(num_hidden, num_classes)))
        self.bias_h = nn.Parameter(torch.zeros(num_hidden))
        self.bias_p = nn.Parameter(torch.zeros(num_classes))

        # hard
        self.device = device

    def forward(self, x):
        """
        """
        X = x
        h_t = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        for t in range(X.size()[1]):
            h_t = self.activation(torch.matmul(X[:, t].view(-1, 1), self.W_hx) + torch.matmul(h_t, self.W_hh) \
                                  + self.bias_h)

        return h_t @ self.W_ph + self.bias_p


if __name__ == '__main__':
    palindrom_generator = PalindromeDataset(3)
    pali = palindrom_generator.generate_palindrome()
    print(pali)

    ############################### Comment out Before Submission #######################
    # defaults
    config = {'model_type': 'RNN', 'seq_length': 5, 'input_length': 10, 'input_dim': 1, 'num_classes': 10,
              'num_hidden': 100, 'batch_size': 128, 'learning_rate': 0.001, 'train_steps': 10000, 'max_norm': 10.0,
              'device': 'cpu'}
    config = Namespace(**config)
    ###########################################################################


    # Initialize the dataset and data loader (note the +1)
    config
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    X_itarable = enumerate(data_loader)
    step, (X, y) = next(X_itarable)

    model = VanillaRNN(**config.__dict__)
    model.forward(X)
