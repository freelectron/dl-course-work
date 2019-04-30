import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle


# ====================================== Plotting Task 1:  RNN ==================================================

path_loss = "RNN_mean_losses.p"
path_acc = "RNN_mean_accuracies.p"

date_time = "2019-04-29-"
dict_loss = pickle.load(open( "RNN_mean_losses.p", "rb" ) )
dict_acc = pickle.load(open( "RNN_mean_accuracies.p", "rb" ) )


# # date_time = datetime.now().replace(second=0, microsecond=0).strftime(format="%Y-%m-%d-%H-%M")
# accuracy_test_log = np.load(os.path.join(path, date_time + "accuracy_test.npy"))
# loss_test_log = np.load(os.path.join(path, date_time + "loss_test.npy"))
# loss_train_log = np.load(os.path.join(path, date_time + "loss_train.npy"))
# accuracy_train_log = np.load(os.path.join(path, date_time + "accuracy_train.npy"))
# #
plt.style.use('ggplot')
fig, ax = plt.subplots()

# grid = range( max(list(map(len, dict_acc.values()))) )

# ####    Plot ACCURACY
for seq_length in dict_acc.keys():
    ax.plot(range(len(dict_acc[seq_length])), dict_acc[seq_length], alpha=0.5, label=seq_length, linewidth=2.0)

ax.set_title('RNN: Accuracy')
ax.legend(loc='upper right', prop={'size': 10})
ax.set_ylabel("Accuracy")
ax.set_xlabel("Iteration x50")

plt.show()
fig.savefig(date_time + 'RNNAccuracyPlot.png')

fig, ax = plt.subplots()

# ####    Plot LOSS
for seq_length in dict_loss.keys():
    ax.plot(range(len(dict_loss[seq_length])), dict_loss[seq_length], alpha=0.7, label=seq_length, linewidth=2.0)

ax.set_title('RNN: Loss')
ax.legend(loc='upper right', prop={'size': 10})
ax.set_ylabel("Loss")
ax.set_xlabel("Iteration x50")

plt.show()
fig.savefig(date_time + 'RNNLossPlot.png')


# ====================================== Plotting Task 2:  LSTM ================================================
path_loss = "LSTM_mean_losses.p"
path_acc = "LSTM_mean_accuracies.p"

date_time = "2019-04-29-"
dict_loss = pickle.load(open( "LSTM_mean_losses.p", "rb" ) )
dict_acc = pickle.load(open( "LSTM_mean_accuracies.p", "rb" ) )


# # date_time = datetime.now().replace(second=0, microsecond=0).strftime(format="%Y-%m-%d-%H-%M")
# accuracy_test_log = np.load(os.path.join(path, date_time + "accuracy_test.npy"))
# loss_test_log = np.load(os.path.join(path, date_time + "loss_test.npy"))
# loss_train_log = np.load(os.path.join(path, date_time + "loss_train.npy"))
# accuracy_train_log = np.load(os.path.join(path, date_time + "accuracy_train.npy"))
# #
plt.style.use('ggplot')
fig, ax = plt.subplots()

# grid = range( max(list(map(len, dict_acc.values()))) )

# ####    Plot ACCURACY
for seq_length in dict_acc.keys():
    ax.plot(range(len(dict_acc[seq_length])), dict_acc[seq_length], alpha=0.5, label=seq_length, linewidth=2.0)

ax.set_title('LSTM: Accuracy')
ax.legend(loc='lower right', prop={'size': 10})
ax.set_ylabel("Accuracy")
ax.set_xlabel("Iteration x50")

plt.show()
fig.savefig(date_time + 'LSTMAccuracyPlot.png')

fig, ax = plt.subplots()

# ####    Plot LOSS
for seq_length in dict_loss.keys():
    ax.plot(range(len(dict_loss[seq_length])), dict_loss[seq_length], alpha=0.7, label=seq_length, linewidth=2.0)

ax.set_title('LSTM: Loss')
ax.legend(loc='upper right', prop={'size': 10})
ax.set_ylabel("Loss")
ax.set_xlabel("Iteration x50")

plt.show()
fig.savefig(date_time + 'LSTMLossPlot.png')

# ====================================== Plotting Task 2:  PyTorch MLP ================================================


# path = "./convnet_results_pytorch/"
# date_time = "2019-04-18-22-32"
# # date_time = datetime.now().replace(second=0, microsecond=0).strftime(format="%Y-%m-%d-%H-%M")
# accuracy_test_log = np.load(os.path.join(path, date_time + "accuracy_test.npy"))
# loss_test_log = np.load(os.path.join(path, date_time + "loss_test.npy"))
# loss_train_log = np.load(os.path.join(path, date_time + "loss_train.npy"))
# accuracy_train_log = np.load(os.path.join(path, date_time + "accuracy_train.npy"))
#
# plt.style.use('ggplot')
#
# # Plot ACCURACY
# grid = list(range(len(loss_test_log)))
#
# fig, ax = plt.subplots()
# ax.plot(grid, accuracy_test_log, alpha=0.5, color='blue', label='test', linewidth=2.0)
# ax.plot(grid, accuracy_train_log, alpha=0.5, color='red', label='train', linewidth=2.0)
#
# ax.set_title('PyTorch ConvNet: Accuracy')
# ax.legend(loc='lower right', prop={'size': 15})
# ax.set_ylabel("Accuracy")
# ax.set_xlabel("Iteration x100")
#
# plt.show()
# fig.savefig(os.path.join(path, date_time + 'accuracyPlotConvNet.png'))
#
# # Plot LOSS
# grid = list(range(len(loss_test_log)))
#
# fig, ax = plt.subplots()
# ax.plot(grid, loss_test_log, alpha=0.5, color='blue', label='test', linewidth=2.0)
# ax.plot(grid, loss_train_log, alpha=0.5, color='red', label='train', linewidth=2.0)
#
# ax.set_title('PyTorch ConvNet: Loss')
# ax.legend(loc='upper right', prop={'size': 15})
# ax.set_ylabel("Loss")
# ax.set_xlabel("Iteration x100")
#
# plt.show()
# fig.savefig(os.path.join(path, date_time + 'lossPlotConvNet.png'))