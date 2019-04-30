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
from dataset import PalindromeDataset


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu', **kwargs):
        super(LSTM, self).__init__()
        self.X_dimensions = (batch_size, seq_length, input_dim)
        self.seq_length = seq_length
        self.input_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device
        self.activation_gates = nn.Sigmoid()  #nn.Tanh()
        self.activation_hidden = nn.Tanh()

        # consist of W_-x matrices
        self.W_x = nn.init.kaiming_normal_(nn.Parameter(torch.rand(4, input_dim, num_hidden)))
        self.W_h = nn.init.kaiming_normal_(nn.Parameter(torch.rand(4, num_hidden, num_hidden)))
        self.W_p = nn.init.kaiming_normal_(nn.Parameter(torch.rand(num_hidden, num_classes)))
        self.bias_gates = nn.Parameter(torch.rand(4, batch_size, num_hidden))
        self.bias_p = nn.Parameter(torch.rand(batch_size, num_classes))

    def forward(self, x):
        """
        x : tensor [Batch_size, seq_length, dim] (e.g. dim = 1)
        """
        ############################## Approach 1 #########################################################
        # X = x
        # h_t = torch.zeros(4, self.batch_size, self.num_hidden, device=self.device)
        # c_tmin1 = torch.zeros(4, self.batch_size, self.num_hidden, device=self.device)
        # for t in range(X.size()[1]):
        #     # create a tensor with input x's dims: [4,B,D)
        #     # X_in  = torch.cat([X[None,:,0,:],X[None,:,0,:],X[None,:,0,:],X[None,:,0,:]],dim=0)
        #     X_t  = torch.cat([X[:, t, :].view(1, -1, 1),X[:, t, :].view(1, -1, 1),X[:, t, :].view(1, -1, 1),X[:, t, :].view(1, -1, 1)],dim=0)
        #     # term1 = torch.einsum('kij,bij->kji', X_in, self.W_x)  # kij,kij->kij , kij,kjl->kil
        #     tmm1 = torch.einsum('kij,kjl->kil', X_t, self.W_x)
        #     tmm2 = torch.einsum('kij,kjl->kil', h_t, self.W_h)
        #     gates_t = self.activation_gates(tmm1 + tmm2 + self.bias_gates)
        #     c_t = gates_t[0,:,:] * gates_t[1,:,:] + c_tmin1 * gates_t[2,:,:]
        #     h_t = self.activation_hidden(c_t) * gates_t[3,:,:]
        #
        # return h_t[0,:,:]

        #####################################################################################################

        X = x
        h_t = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        c_t = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        for t in range(X.size()[1]):
            # create a tensor with input x's dims: [4,B,D)
            X_t = torch.cat([X[:, t, :].view(1, -1, 1), X[:, t, :].view(1, -1, 1),X[:, t, :].view(1, -1, 1),X[:, t, :].view(1, -1, 1)], dim=0)
            # term1 = torch.einsum('kij,bij->kji', X_in, self.W_x)  # kij,kij->kij , kij,kjl->kil
            tmm1 = torch.einsum('kij,kjl->kil', X_t, self.W_x)
            tmm2 = torch.einsum('ij,kjl->kil', h_t, self.W_h)
            gates_t = self.activation_gates(tmm1 + tmm2 + self.bias_gates)
            c_t = gates_t[0, :, :] * gates_t[1, :, :] + c_t * gates_t[2, :, :]
            h_t = self.activation_hidden(c_t) * gates_t[3, :, :]

        return torch.matmul(h_t, self.W_p) + self.bias_p


if __name__ == '__main__':
    palindrom_generator = PalindromeDataset(3)
    pali = palindrom_generator.generate_palindrome()
    print(pali)

    ############################### Comment out Before Submission #######################
    # defaults
    config = {'model_type': 'LSTM', 'seq_length': 5, 'input_length': 10, 'input_dim': 1, 'num_classes': 10,
              'num_hidden': 100, 'batch_size': 128, 'learning_rate': 0.001, 'train_steps': 10000, 'max_norm': 10.0,
              'device': 'cpu'}
    config = Namespace(**config)
    ###########################################################################


    # Initialize the dataset and data loader (note the +1)
    config
    dataset = PalindromeDataset(config.input_length + 1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    model = LSTM(**config.__dict__)

    X_itarable = enumerate(data_loader)
    step, (X, y) = next(X_itarable)
    if X.dim() != len(model.X_dimensions):
        X = X.view(X.size()[0], X.size()[1], 1) #X[..., None]


    model.forward(X)
