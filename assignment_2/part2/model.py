# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import argparse


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from part2.dataset import TextDataset


class LSTM_take_output(nn.Module):
    """
    Make nn.Sequential work with LSTM's by selecting the correct output of last
    """

    def __init__(self, vocabulary_size, lstm_num_hidden, lstm_num_layers, batch_first):
        super(LSTM_take_output,self).__init__()

        self.lstm_layers = nn.LSTM(input_size=vocabulary_size,
                                   hidden_size=lstm_num_hidden,
                                   num_layers=lstm_num_layers,
                                   batch_first=True)

        self.model = nn.Sequential(self.lstm_layers)

    def forward(self, x):
        """
        """
        return self.model.forward(x)


class TextGenerationModel(nn.Module):
    """
    In order to use nn.Sequential use LSTM_take_output
    """

    def __init__(self, batch_size, seq_length, vocabulary_size, lstm_num_hidden, lstm_num_layers, device, **kwargs):
        super(TextGenerationModel, self).__init__()
        self.X_dimensions = (batch_size, seq_length, vocabulary_size)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.device = device

        # create lstm layers
        self.layers = list()
        # self.lstm_layers = nn.LSTM(input_size=vocabulary_size,
        #                            hidden_size=lstm_num_hidden,
        #                            num_layers=lstm_num_layers,
        #                            batch_first=True)

        self.lstm = LSTM_take_output(vocabulary_size=vocabulary_size,
                                            lstm_num_hidden=lstm_num_hidden,
                                            lstm_num_layers=lstm_num_layers,
                                            batch_first=True)

        # map to characters with linear layers
        self.layers.append(nn.Linear(in_features=lstm_num_hidden,
                                     out_features=vocabulary_size,
                                     bias=True))
        # create model
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        """
        out, (h_t, c_t) = self.lstm.forward(x)
        return self.model.forward(out)

if __name__ == '__main__':
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str,  default='./part2/een_klein_heldendicht.txt', required=False, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()


    dataset = TextDataset(config.txt_file, config.seq_length)  #'./part2/een_klein_heldendicht.txt', 10)

    # get a couple of sequance examples from batches
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # for step, (batch_inputs, batch_targets) in enumerate(data_loader):
    X_itarable = enumerate(data_loader)
    step, (X_transposed, y_transposed) = next(X_itarable)
    X_batch = torch.stack(X_transposed).t()
    Y_batch = torch.stack(y_transposed).t()

    # one-hot encode
    X = torch.zeros(len(X_batch), 30, dataset.vocab_size).scatter_(2, X_batch.unsqueeze(2), 1)

    #     X = batch_inputs.to(device)
    #     y = batch_targets.to(device)

    model = TextGenerationModel(vocabulary_size=dataset.vocab_size, device='cpu', **config.__dict__)

    model.forward(X)
