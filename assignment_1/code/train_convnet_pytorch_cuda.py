"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

import os
import logging
from datetime import datetime

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from convnet_pytorch import ConvNet
import cifar10_utils

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 100
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
FLAGS = None
logging.getLogger().setLevel(logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info('You are using %s', device)

# Handy to have all tensors with the same type
dtype = torch.cuda.FloatTensor


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    accuracy = (predictions.argmax(dim=1) == targets.argmax(dim=1)).type(dtype).mean()

    return accuracy.item()


def train():
    """
    Performs training and evaluation of ConvNet model.

    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    # ## Prepare all functions
    # # Get number of units in each hidden layer specified in the string such as 100,100
    # if FLAGS.dnn_hidden_units:
    #     dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    #     dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    # else:
    #     dnn_hidden_units = []

    # Set path to data
    data_dir = FLAGS.data_dir

    data = cifar10_utils.get_cifar10(data_dir)

    # Prepare the test set
    input_dims_test = data['test'].images.shape
    height = input_dims_test[2]
    width = input_dims_test[3]
    channels = input_dims_test[1]
    # num_images_test = input_dims_test[0]
    # image_dims_ravel = height * width * channels

    X_test = data["test"].images
    Y_test = data["test"].labels

    # Make acceptable input for test
    # X_test = X_test.reshape((num_images_test, image_dims_ravel))

    # make usable by pytorch
    X_test = torch.tensor(X_test, requires_grad=False).type(dtype).to(device)
    Y_test = torch.tensor(Y_test, requires_grad=False).type(dtype).to(device)

    # Determine the channels
    n_channels = channels

    model = ConvNet(n_channels=n_channels, n_classes=10)
    model.cuda()

    accuracy_train_log = list()
    accuracy_test_log = list()
    loss_train_log = list()
    loss_test_log = list()

    # FLAGS hold command line arguments
    batch_size = FLAGS.batch_size
    numb_iterations = FLAGS.max_steps
    learning_rate = FLAGS.learning_rate
    evaluation_freq = FLAGS.eval_freq
    logging.info(f"learning rate: %2d " % learning_rate)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # X_train = data['train'].images.reshape((data['train'].images.shape[0],
    #                                         image_dims_ravel))

    X_train = data['train'].images
    Y_train = data['train'].labels

    X_train = torch.tensor(X_train, requires_grad=False).type(dtype).to(device)
    Y_train = torch.tensor(Y_train, requires_grad=False).type(dtype).to(device)

    targs_train = Y_train.argmax(dim=1)

    # running_loss = loss_current.detach().item()

    for step in range(numb_iterations):

        X_batch, Y_batch = data['train'].next_batch(batch_size)

        # X_batch = X_batch.reshape((batch_size, image_dims_ravel))

        # Convert to tensors which are handled by the device
        X_batch = torch.from_numpy(X_batch).type(dtype).to(device)
        Y_batch = torch.from_numpy(Y_batch).type(dtype).to(device)

        # why do we need this again?
        optimizer.zero_grad()

        targs = Y_batch.argmax(dim=1)
        outputs = model.forward(X_batch)
        loss_current = criterion(outputs, targs)
        loss_current.backward()
        optimizer.step()

        running_loss = loss_current.detach().item()

        if step % evaluation_freq == 0:
            list_acc = list()
            list_loss = list()
            for i in range(0, 70):
                selection = random.sample(range(1, 5000), 64)
                targs_train = Y_train[selection].argmax(dim=1)
                outputs_train = model(X_train[selection])
                loss_current_train = criterion(outputs_train, targs_train).detach().item()
                acc_current_train = accuracy(outputs_train, Y_train[selection])
                list_loss.append(loss_current_train)
                list_acc.append(acc_current_train)
            loss_train_log.append(np.mean(list_loss))
            accuracy_train_log.append(np.mean(list_acc))
            logging.info(f"train performance: loss = %4f, accuracy = %4f ", loss_train_log[-1], accuracy_train_log[-1])

            # loss_train_log.append(running_loss)
            # accuracy_train_log.append(accuracy(outputs, Y_batch))
            # logging.info(f"train performance: loss = %4f, accuracy = %4f ", loss_train_log[-1], accuracy_train_log[-1])

            # Get performance on the test set
           # targs_test = Y_test.argmax(dim=1)
           # outputs_test = model(X_test)
           # test_loss_current = criterion(outputs_test, targs_test).detach().item()
            list_acc = list()
            list_loss= list()
            for i in range(0, 15):
                selection = random.sample(range(1, 1000), 64)
                targs_test = Y_test[selection].argmax(dim=1)
                outputs_test = model(X_test[selection])
                loss_current_test = criterion(outputs_test, targs_test).detach().item()
                acc_current_test = accuracy(outputs_test, Y_test[selection])
                list_loss.append(loss_current_test)
                list_acc.append(acc_current_test)
            loss_test_log.append(np.mean(list_loss))
            accuracy_test_log.append(np.mean(list_acc))
            logging.info(f"test performance: loss = %4f , accuracy = %4f\n", loss_test_log[-1], accuracy_test_log[-1])

            # TODO: implement early stopping ?

    path = "./convnet_results_pytorch/"
    date_time = datetime.now().replace(second=0, microsecond=0).strftime(format="%Y-%m-%d-%H-%M")
    np.save(os.path.join(path, date_time + "accuracy_test"), accuracy_test_log)
    np.save(os.path.join(path, date_time + "loss_test"), loss_test_log)
    np.save(os.path.join(path, date_time + "loss_train"), loss_train_log)
    np.save(os.path.join(path, date_time + "accuracy_train"), accuracy_train_log)


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')

    FLAGS, unparsed = parser.parse_known_args()

    main()
