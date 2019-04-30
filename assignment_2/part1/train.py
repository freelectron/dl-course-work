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

import argparse
from argparse import Namespace
import time
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter
dtype = torch.FloatTensor

################################################################################

def train(config, acc_th=0.99, epsilon=0.01):
    """
    """
    # some additional vars
    learning_rate = config.learning_rate

    # input_length = seq_length (?)
    seq_length = config.input_length

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        # Because I added kwargs to VanillaRNN, this will work
        model = VanillaRNN(seq_length, **config.__dict__)
    else:
        model = LSTM(seq_length, **config.__dict__)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss=CrossEntropy and optimizer=SGD
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    loss_list = list()
    accuracy_list = list()

    mean_loss_list = list()
    mean_accuracy_list = list()

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # X_batch = torch.stack(X_transposed).t()
        # Y_batch = torch.stack(y_transposed).t()
        X = batch_inputs.to(device)
        y = batch_targets.to(device)

        if config.model_type == 'LSTM':
            if X.dim() != len(model.X_dimensions):
                X = X.view(X.size()[0], X.size()[1], 1)

        model.forward(X)

        # TODO: Try one-hot encode
        # make X to be (B, L, D) where D is 10 (one-hot encoded)
        #torch.zeros(len(x), x.max() + 1).scatter_(1, x.unsqueeze(1), 1.)

        # Only for time measurement of step through network
        t1 = time.time()

        # Add more code here ...
        optimizer.zero_grad()
        outputs = model.forward(X)

        # Add more code here ...
        loss_current = criterion(outputs, y)
        loss_current.backward(retain_graph=True)
        optimizer.step()

        ############################################################################
        # QUESTION: what happens here and why?   - RESCALING ?
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        loss = loss_current.detach().item()
        accuracy = (outputs.argmax(dim=1) == y.long()).sum().float() / float(y.shape[0])

        loss_list.append(loss)
        accuracy_list.append(accuracy)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 50 == 0:

            mean_loss_list.append(np.mean(loss_list[-50:]))
            mean_accuracy_list.append(np.mean(accuracy_list[-50:]))

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

            if step == config.train_steps or mean_loss_list[-1] < epsilon: #or mean_accuracy_list[-1] > acc_th:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                print(mean_loss_list[-1])
                print(mean_accuracy_list[-1])
                break

    print('Done training.')
    return mean_loss_list, mean_accuracy_list


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # ############################### Comment out Before Submission #######################
    # defaults
    # config = {'model_type': 'LSTM', 'input_length': 13, 'input_dim': 1, 'num_classes': 10, 'num_hidden': 128,
    #           'batch_size': 128, 'learning_rate': 0.001, 'train_steps': 10000, 'max_norm': 10.0, 'device': 'cpu'}
    # config = Namespace(**config)
    ###########################################################################

    # ############################### Comment out Before Submission #######################
    # # defaults
    # config = {'model_type': 'LSTM', 'input_length': 5, 'input_dim': 1, 'num_classes': 10, 'num_hidden': 100,
    #           'batch_size': 128, 'learning_rate': 0.001, 'train_steps': 10000, 'max_norm': 10.0, 'device': 'cpu'}
    # config = Namespace(**config)
    # ###########################################################################

    train(config)

    # ======================================  RNN ==================================================

    # dict_losses = dict()
    # dict_accuracies = dict()
    #
    # for seq_legnth in [5, 7, 10, 12, 15, 17, 20, 25]:
    #     config = {'model_type': 'RNN', 'input_length': seq_legnth, 'input_dim': 1, 'num_classes': 10, 'num_hidden': 128,
    #               'batch_size': 128, 'learning_rate': 0.001, 'train_steps': 10000, 'max_norm': 10.0, 'device': 'cpu'}
    #     config = Namespace(**config)
    #
    #     mean_loss, mean_accuracy = train(config)
    #
    #     dict_losses[f'input_length_{seq_legnth}'] = mean_loss
    #     dict_accuracies[f'input_length_{seq_legnth}'] = mean_accuracy
    #
    #
    #
    # import pickle
    #
    # pickle.dump(dict_losses, open("RNN_mean_losses.p", "wb"))
    # pickle.dump(dict_accuracies, open("RNN_mean_accuracies.p", "wb"))

    # ======================================  LSTM ================================================================

    # dict_losses = dict()
    # dict_accuracies = dict()
    #
    # for seq_legnth in [5, 7, 10, 12, 15, 17, 20, 25]:
    #     config = {'model_type': 'LSTM', 'input_length': seq_legnth, 'input_dim': 1, 'num_classes': 10, 'num_hidden': 128,
    #               'batch_size': 128, 'learning_rate': 0.001, 'train_steps': 10000, 'max_norm': 10.0, 'device': 'cpu'}
    #     config = Namespace(**config)
    #
    #     mean_loss, mean_accuracy = train(config)
    #
    #     dict_losses[f'input_length_{seq_legnth}'] = mean_loss
    #     dict_accuracies[f'input_length_{seq_legnth}'] = mean_accuracy
    #
    #
    # import pickle
    #
    # pickle.dump(dict_losses, open("LSTM_mean_losses.p", "wb"))
    # pickle.dump(dict_accuracies, open("LSTM_mean_accuracies.sp", "wb"))
