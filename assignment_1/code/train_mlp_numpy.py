"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
from datetime import datetime

import argparse
import numpy as np
import os

import cifar10_utils
from mlp_numpy import MLP
from modules import CrossEntropyModule

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
FLAGS = None
logging.getLogger().setLevel(logging.INFO)


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

    Implement accuracy computation.
    """

    accuracy = (predictions.argmax(axis=1) == targets.argmax(axis=1)).mean()

    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.

    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    data_dir = FLAGS.data_dir

    data = cifar10_utils.get_cifar10(data_dir)

    # Prepare the test set
    input_dims_test = data['test'].images.shape
    height = input_dims_test[1]
    width = input_dims_test[2]
    channels = input_dims_test[3]
    num_images_test = input_dims_test[0]
    image_dims_ravel = height * width * channels

    X_test = data["test"].images
    y_test = data["test"].labels
    # Make acceptable input for test
    X_test = X_test.reshape((num_images_test, image_dims_ravel))

    # Create model
    model = MLP(n_inputs=image_dims_ravel, n_hidden=dnn_hidden_units, n_classes=10)

    # Before backprop calc loss and its derivative
    loss = CrossEntropyModule()

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

    for step in range(numb_iterations):

        X, y = data['train'].next_batch(batch_size)

        X = X.reshape((batch_size, image_dims_ravel))

        output = model.forward(X)

        loss_model_current = loss.forward(output, y)
        dL_dxN = loss.backward(output, y)

        model.backward(dL_dxN)

        # Update weights
        for module in model.linear_modules:
            module.params["weight"] -= learning_rate * module.grads["weight"]
            module.params["bias"] -= learning_rate * module.grads["bias"]

        if step % evaluation_freq == 0:
            # evaluate on the whole train set, not only on the bathes
            output = model.forward(data['train'].images.reshape((data['train'].images.shape[0],
                                                                 image_dims_ravel)))
            loss_model_current = loss.forward(output, data['train'].labels)
            loss_train_log.append(loss_model_current)
            accuracy_train_log.append(accuracy(output, data['train'].labels))
            logging.info(f"train performance: loss = %4f, accuracy = %4f ", loss_train_log[-1], accuracy_train_log[-1])

            output = model.forward(X_test)
            loss_test_log.append(loss.forward(output, y_test))
            accuracy_test_log.append(accuracy(output, y_test))
            logging.info(f"test performance: loss = %4f , accuracy = %4f", loss_test_log[-1], accuracy_test_log[-1])

    path = "./mlp_results_numpy/"
    date_time = datetime.now().replace(second=0, microsecond=0).strftime(format="%Y-%m-%d-%H-%M")
    np.save(os.path.join(path, date_time + "accuracy_test"), accuracy_test_log)
    np.save(os.path.join(path, date_time + "loss_test"), loss_test_log)
    np.save(os.path.join(path, date_time + "loss_train"), loss_train_log)
    np.save(os.path.join(path, date_time + "accuracy_train"), accuracy_train_log)

    logging.info("SAVED in %s", path)

    return accuracy_test_log


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

    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
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
