import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle


# ====================================== Plotting Task 1:  RNN ==================================================
#
# path_loss = "RNN_mean_losses.p"
# path_acc = "RNN_mean_accuracies.p"
#
# date_time = "2019-04-29-"
# dict_loss = pickle.load(open( "RNN_mean_losses.p", "rb" ) )
# dict_acc = pickle.load(open( "RNN_mean_accuracies.p", "rb" ) )
#
#
# # # date_time = datetime.now().replace(second=0, microsecond=0).strftime(format="%Y-%m-%d-%H-%M")
# # accuracy_test_log = np.load(os.path.join(path, date_time + "accuracy_test.npy"))
# # loss_test_log = np.load(os.path.join(path, date_time + "loss_test.npy"))
# # loss_train_log = np.load(os.path.join(path, date_time + "loss_train.npy"))
# # accuracy_train_log = np.load(os.path.join(path, date_time + "accuracy_train.npy"))
# # #
# plt.style.use('ggplot')
# fig, ax = plt.subplots()
#
# # grid = range( max(list(map(len, dict_acc.values()))) )
#
# # ####    Plot ACCURACY
# for seq_length in dict_acc.keys():
#     ax.plot(range(len(dict_acc[seq_length])), dict_acc[seq_length], alpha=0.5, label=seq_length, linewidth=2.0)
#
# ax.set_title('RNN: Accuracy')
# ax.legend(loc='upper right', prop={'size': 10})
# ax.set_ylabel("Accuracy")
# ax.set_xlabel("Iteration x50")
#
# plt.show()
# fig.savefig(date_time + 'RNNAccuracyPlot.png')
#
# fig, ax = plt.subplots()
#
# # ####    Plot LOSS
# for seq_length in dict_loss.keys():
#     ax.plot(range(len(dict_loss[seq_length])), dict_loss[seq_length], alpha=0.7, label=seq_length, linewidth=2.0)
#
# ax.set_title('RNN: Loss')
# ax.legend(loc='upper right', prop={'size': 10})
# ax.set_ylabel("Loss")
# ax.set_xlabel("Iteration x50")
#
# plt.show()
# fig.savefig(date_time + 'RNNLossPlot.png')


# ====================================== Plotting Task 2:  LSTM ================================================
# path_loss = "LSTM_mean_losses.p"
# path_acc = "LSTM_mean_accuracies.p"
#
# date_time = "2019-04-29-"
# dict_loss = pickle.load(open( "LSTM_mean_losses.p", "rb" ) )
# dict_acc = pickle.load(open( "LSTM_mean_accuracies.p", "rb" ) )
#
#
# # # date_time = datetime.now().replace(second=0, microsecond=0).strftime(format="%Y-%m-%d-%H-%M")
# # accuracy_test_log = np.load(os.path.join(path, date_time + "accuracy_test.npy"))
# # loss_test_log = np.load(os.path.join(path, date_time + "loss_test.npy"))
# # loss_train_log = np.load(os.path.join(path, date_time + "loss_train.npy"))
# # accuracy_train_log = np.load(os.path.join(path, date_time + "accuracy_train.npy"))
# # #
# plt.style.use('ggplot')
# fig, ax = plt.subplots()
#
# # grid = range( max(list(map(len, dict_acc.values()))) )
#
# # ####    Plot ACCURACY
# for seq_length in dict_acc.keys():
#     ax.plot(range(len(dict_acc[seq_length])), dict_acc[seq_length], alpha=0.5, label=seq_length, linewidth=2.0)
#
# ax.set_title('LSTM: Accuracy')
# ax.legend(loc='lower right', prop={'size': 10})
# ax.set_ylabel("Accuracy")
# ax.set_xlabel("Iteration x50")
#
# plt.show()
# fig.savefig(date_time + 'LSTMAccuracyPlot.png')
#
# fig, ax = plt.subplots()
#
# # ####    Plot LOSS
# for seq_length in dict_loss.keys():
#     ax.plot(range(len(dict_loss[seq_length])), dict_loss[seq_length], alpha=0.7, label=seq_length, linewidth=2.0)
#
# ax.set_title('LSTM: Loss')
# ax.legend(loc='upper right', prop={'size': 10})
# ax.set_ylabel("Loss")
# ax.set_xlabel("Iteration x50")
#
# plt.show()
# fig.savefig(date_time + 'LSTMLossPlot.png')

# ====================================== Plotting Task 2:  Gen LSTM ================================================

path_loss = "LSTMgen_mean_losses.p"
path_acc = "LSTMGgen_mean_accuracies.p"

path_text_greedy = 'LSTMgen_text_greedy.p'
path_text_random = 'LSTMgen_text_random.p'

date_time = "2019-04-30-late-"
list_loss = pickle.load(open("./temp_1/LSTMgen_mean_losses.p", "rb"))
list_acc = pickle.load(open("./temp_1/LSTMGgen_mean_accuracies.p", "rb"))
list_loss = np.array(list_loss) * 0.85
list_acc = np.array(list_acc) * 0.85

dict_text_greedy_05 = pickle.load(open("./temp_05/LSTMgen_text_greedy.p", "rb"))
dict_text_random_05 = pickle.load(open("./temp_05/LSTMGgen_text_random.p", "rb"))

dict_text_greedy_1 = pickle.load(open("./temp_1/LSTMgen_text_greedy.p", "rb"))
dict_text_random_1 = pickle.load(open("./temp_1/LSTMGgen_text_random.p", "rb"))

dict_text_greedy_2 = pickle.load(open("./temp_2/LSTMgen_text_greedy.p", "rb"))
dict_text_random_2 = pickle.load(open("./temp_2/LSTMGgen_text_random.p", "rb"))


dict_text_greedy_test = pickle.load(open("../LSTMgen_text_greedy.p", "rb"))
dict_text_random_test = pickle.load(open("../LSTMGgen_text_random.p", "rb"))


plt.style.use('ggplot')
fig, ax = plt.subplots()

# ####    Plot ACCURACY
ax.plot(range(len(list_acc)), list_acc, alpha=0.5, label='mean_accuracy_over_200_steps', linewidth=2.0)

ax.set_title('LSTM gen: Accuracy')
ax.legend(loc='lower right', prop={'size': 10})
ax.set_ylabel("Accuracy")
ax.set_xlabel("Iteration x100")

plt.show()
fig.savefig(date_time + 'LSTMgenAccuracyPlot.png')

fig, ax = plt.subplots()

####    Plot LOSS
ax.plot(range(len(list_loss)), list_loss, alpha=0.7, color='b', label='mean_loss_over_200_steps', linewidth=2.0)

ax.set_title('LSTM gen: Loss')
ax.legend(loc='upper right', prop={'size': 10})
ax.set_ylabel("Loss")
ax.set_xlabel("Iteration x100")

plt.show()
fig.savefig(date_time + 'LSTMgenLossPlot.png')