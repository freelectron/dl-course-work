import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ====================================== Plotting Task 1:  Numpy MLP ==================================================

# path = "./mlp_results_numpy/"
# date_time = "2019-04-18-21-18"
# # date_time = datetime.now().replace(second=0, microsecond=0).strftime(format="%Y-%m-%d-%H-%M")
# accuracy_test_log = np.load(os.path.join(path, date_time + "accuracy_test.npy"))
# loss_test_log = np.load(os.path.join(path, date_time + "loss_test.npy"))
# loss_train_log = np.load(os.path.join(path, date_time + "loss_train.npy"))
# accuracy_train_log = np.load(os.path.join(path, date_time + "accuracy_train.npy"))
#
# plt.style.use('ggplot')
#
# ####    Plot ACCURACY
# grid = list(range(len(loss_test_log)))
#
# fig, ax = plt.subplots()
# ax.plot(grid, accuracy_test_log, alpha=0.5, color='blue', label='test', linewidth=2.0)
# ax.plot(grid, accuracy_train_log, alpha=0.5, color='red', label='train', linewidth=2.0)
#
# ax.set_title('MLP Numpy: Accuracy')
# ax.legend(loc='lower right', prop={'size': 15})
# ax.set_ylabel("Accuracy")
# ax.set_xlabel("Iteration x100")
#
# plt.show()
# fig.savefig(os.path.join(path, date_time + 'accuracyPlot.png'))
#
# ####    Plot LOSS
# grid = list(range(len(loss_test_log)))
#
# fig, ax = plt.subplots()
# ax.plot(grid, loss_test_log, alpha=0.5, color='blue', label='test', linewidth=2.0)
# ax.plot(grid, loss_train_log, alpha=0.5, color='red', label='train', linewidth=2.0)
#
# ax.set_title('MLP Numpy: Loss')
# ax.legend(loc='upper right', prop={'size': 15})
# ax.set_ylabel("Loss")
# ax.set_xlabel("Iteration x100")
#
# plt.show()
# fig.savefig(os.path.join(path, date_time + 'lossPlot.png'))


# ====================================== Plotting Task 2:  PyTorch MLP ================================================
#
# path = "./mlp_results_pytorch/"
# # date_time = "2019-04-19-10-42"
# date_time = "2019-04-19-13-29"
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
# # ax.plot(grid, accuracy_test_log, alpha=0.5, color='blue', label='test', linewidth=2.0)
# # ax.plot(grid, accuracy_train_log, alpha=0.5, color='red', label='train', linewidth=2.0)
#
# ax.set_title('PyTorch MLP: Accuracy')
# ax.legend(loc='lower right', prop={'size': 15})
# ax.set_ylabel("Accuracy")
# ax.set_xlabel("Iteration x100")
#
# plt.show()
# fig.savefig(os.path.join(path, date_time + 'accuracyPlot.png'))
# #
# # # Plot LOSS
# grid = list(range(len(loss_test_log)))
# #
# fig, ax = plt.subplots()
# # ax.plot(grid, loss_test_log, alpha=0.5, color='blue', label='test', linewidth=2.0)
# # ax.plot(grid, loss_train_log, alpha=0.5, color='red', label='train', linewidth=2.0)
#
# ax.set_title('PyTorch MLP: Loss')
# ax.legend(loc='upper right', prop={'size': 15})
# ax.set_ylabel("Loss")
# ax.set_xlabel("Iteration x100")
#
# plt.show()
# fig.savefig(os.path.join(path, date_time + 'lossPlot.png'))

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