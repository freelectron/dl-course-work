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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from part2.dataset import TextDataset
from part2.model import TextGenerationModel

dtype = torch.FloatTensor

def text_generator(model, seq_length, temperature, dataset, device):
    """Generates text"""
    greedy_char_indices = list()
    random_char_indices = list()

    with torch.no_grad():
        # First character is generated at random
        char_first = torch.randint(low=0,
                                   high=dataset.vocab_size,
                                   size=(1, 1),
                                   dtype=torch.long,
                                   device=device)

        greedy_char_indices.append(char_first.item())
        random_char_indices.append(char_first.item())

        for i in range(seq_length - 1):
            # 'Greedy' generation
            X = torch.tensor(greedy_char_indices)
            X = torch.zeros(len(greedy_char_indices), dataset.vocab_size).scatter_(1, X.unsqueeze(1), 1)
            X = X.view(1, len(greedy_char_indices), dataset.vocab_size)
            y = model(X)
            generated_char_idx = y[:, i, :].argmax()
            greedy_char_indices.append(generated_char_idx.item())

            # Sample with temperature
            X = torch.tensor(random_char_indices)
            X = torch.zeros(len(random_char_indices), dataset.vocab_size).scatter_(1, X.unsqueeze(1), 1)
            X = X.view(1, len(random_char_indices), dataset.vocab_size)
            y = model(X)

            probs = torch.softmax(y[:, i, :].squeeze() / temperature, dim=0)
            generated_char_idx = torch.multinomial(probs, 1).item()
            random_char_indices.append(generated_char_idx)

    # TODO: report results for the following temperatures : T ∈ {0.5, 1.0, 2.0}
    text_greedy = dataset.convert_to_string(greedy_char_indices)
    text_random = dataset.convert_to_string(random_char_indices)

    return text_greedy, text_random

def train(config):
    """
    """

    # some additional vars
    learning_rate = config.learning_rate

    # TODO: Initialize the device which to run the model on
    device = 'cpu'
    device = torch.device(device)

    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(vocabulary_size=dataset.vocab_size, device='cpu', **config.__dict__)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    # evaluation
    loss_list = list()
    accuracy_list = list()

    mean_loss_list = list()
    mean_accuracy_list = list()

    step = 0
    epoch = 0
    steps_total = 0

    while steps_total < config.train_steps:
        epoch += 1
        for step, (X_transposed, y_transposed) in enumerate(data_loader):
            steps_total = step * epoch
            # Only for time measurement of step through network
            t1 = time.time()

            X_batch = torch.stack(X_transposed).t()
            Y_batch = torch.stack(y_transposed).t()

            X = X_batch.to(device)
            y = Y_batch.to(device)

            X = torch.zeros(len(X), config.seq_length, dataset.vocab_size).scatter_(2, X.unsqueeze(2), 1)

            optimizer.zero_grad()
            outputs = model.forward(X).type(dtype)

            # Add more code here ...
            loss_current = criterion(outputs.transpose(2, 1), y)
            loss_current.backward(retain_graph=True)
            optimizer.step()

            # evaluation
            loss = loss_current.detach().item()
            accuracy = (outputs.argmax(dim=2) == y.long()).sum().float() / (float(y.shape[0]) * float(y.shape[1]))

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            loss_list.append(loss)
            accuracy_list.append(accuracy)

            text_greedy_generated = dict()
            text_random_generated = dict()

            if step % config.print_every == 0:

                mean_loss_list.append(np.mean(loss_list[-50:]))
                mean_accuracy_list.append(np.mean(accuracy_list[-50:]))

                print("[{}] Train Step {}/{}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), steps_total,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                ))

                # Text generation
                if step % config.sample_every == 0:
                    # Generate some sentences by sampling from the model
                    text_greedy, text_random = text_generator(model, config.seq_length, 0.2, dataset, device)
                    text_greedy_generated[len(mean_accuracy_list)] = text_greedy
                    text_random_generated[len(mean_accuracy_list)] = text_random
                    print(text_greedy, len(text_greedy))
                    print(text_random, len(text_random))

                # if step == config.train_steps:
                #     # If you receive a PyTorch data-loader error, check this bug report:
                #     # https://github.com/pytorch/pytorch/pull/9655
                    if step > config.train_steps:
                        break

    print('Done training.')
    return mean_loss_list, mean_accuracy_list, text_greedy_generated, text_random_generated

if __name__ == "__main__":

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

    parser.add_argument('--train_steps', type=int, default=1e4, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=20, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=200, help='How often to sample from the model')

    config = parser.parse_args()

    import pickle

    # Train the model
    mean_loss, mean_accuracy, text_greedy_generated, text_random_generated = train(config)

    pickle.dump(mean_loss, open("LSTMgen_mean_losses.p", "wb"))
    pickle.dump(mean_accuracy, open("LSTMGgen_mean_accuracies.p", "wb"))

    pickle.dump(text_greedy_generated, open("LSTMgen_text_greedy.p", "wb"))
    pickle.dump(text_random_generated, open("LSTMGgen_text_random.p", "wb"))