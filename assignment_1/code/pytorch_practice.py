# practice pytorch from https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
from __future__ import print_function
import torch


# x = torch.empty(5, 3)
#
# x = torch.rand(5, 3) * 10
#
# # construct tensors
# x = torch.tensor([5, 3, 3, 2.3])
#
# x = x.new_ones(5, 3, dtype=torch.double)
# # new tensor of the same size as x, but with random values
# x = torch.randn_like(x)
#
# x = torch.tensor([[5,3,2,9.3],[4,4,5,0.7]])
# y = torch.tensor([[1,1,2,9.3],[1,1,2,0.7]])
# print(x)
# print(y)
#
# # operations
# print(x.size())
#
# result = torch.add(x, y)
# print('result', result)
# # or alternatively (does not matter what size you specify)
# out = torch.empty(size=(5, 4))
# torch.add(x, y, out=out)
# print('out', out)
# # or
# print(x.add_(y))
# print(out[0, 2] * 2)
#
# # resizing
# print(out.view(8))
# print(out.view(4, 2))
# print(out.view(2, 2, 2))

# -------------------- AUTOGRAD ----------------------------
def show_info():
    print(""
          " torch.Tensor is the central class of the package. If you set its attribute .requires_grad as True, it starts \
    to track all operations on it. When you finish your computation you can call .backward() and have all the gradients\
    computed automatically. The gradient for this tensor will be accumulated into .grad attribute.\
    \
    There’s one more class which is very important for autograd implementation - a Function.\
    Tensor and Function are interconnected and build up an acyclic graph, that encodes a complete history of computation.\
    Each tensor has a .grad_fn attribute that references a Function that has created the Tensor (except for Tensors \
    created by the user - their grad_fn is None)."
          "")


# x = torch.ones(2, 2, requires_grad=True)
# print(x)
#
# # tensor operation: element wise multiplication
# # So we need to have both dL_dx, x with the same dtype
# dL_dx = torch.tensor([[2.0, 1], [0, 2]])
# y = dL_dx * x # + 2
# # y was created as a result of an operation, so it has a grad_fn (meaning it got connected to the graph).
# # print(y)
#
# # loss is usually defined to be a scalar so we need to assume y is our loss and do
# y.sum().backward()
# print('dL/dx', x.grad)
#
# print("\n dot product now: \n")
# # tensor operation: dot product
# # So we need to have both dL_dx, x with the same dtype
# dL_dx = torch.tensor([[2.0, 1], [0, 2]])
# y = torch.mm(dL_dx, x)  #+ torch.tensor([[2, 2], [1.0, 1]])
# # y was created as a result of an operation, so it has a grad_fn (meaning it got connected to the graph).
# # print(y)
#
# # loss is usually defined to be a scalar so we need to assume y is our loss and do
# y.sum().backward()
# print('dL/dx', x.grad)


# -------------- Practice with SUPER ---------------------------------

import logging
#
# logging.getLogger().setLevel(logging.INFO)
# class LoggingDict(dict):
#     def __setitem__(self, key, value):
#         logging.info('Settingto:%s;', value)
#         super().__setitem__(key, value)
#
# test = LoggingDict()
#
# test[1] = '33'


# --------------- model params -----------------------------

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
