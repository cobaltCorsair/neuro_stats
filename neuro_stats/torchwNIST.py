import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
import numpy as np

#train = torchvision.datasets.MNIST('./data', train=True, download=True)
#trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=100)


# Load MNIST data
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,), (128,)),
])
train = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
test = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=100)
#testloader = torch.utils.data.DataLoader(test, shuffle=True, batch_size=100)

fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)
for i in range(4):
    for j in range(4):
        ax[i][j].imshow(train.data[4 * i + j], cmap="gray")
plt.show()

#print(train.shape)
print(train.data.shape, train.targets.shape)

X_train = train.data.reshape(-1, 784).float() / 255.0
#X_train = train.data.reshape(-1, 196).float() / 255.0
y_train = train.targets

loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=100)


class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 784)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(784, 10)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.layer2(x)
        return x


if True:
    model = Baseline()

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=100)

    n_epochs = 10
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        #y_pred = model(X_test)
        #acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
        #print("Epoch %d: model accuracy %.2f%%" % (epoch, acc * 100))

    print(X_train.shape)


# class LeNet5(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
#         self.act1 = nn.Tanh()
#         self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
#
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
#         self.act2 = nn.Tanh()
#         self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
#
#         self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
#         self.act3 = nn.Tanh()
#
#         self.flat = nn.Flatten()
#         self.fc1 = nn.Linear(1 * 1 * 120, 84)
#         self.act4 = nn.Tanh()
#         self.fc2 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         # input 1x28x28, output 6x28x28
#         x = self.act1(self.conv1(x))
#         # input 6x28x28, output 6x14x14
#         x = self.pool1(x)
#         # input 6x14x14, output 16x10x10
#         x = self.act2(self.conv2(x))
#         # input 16x10x10, output 16x5x5
#         x = self.pool2(x)
#         # input 16x5x5, output 120x1x1
#         x = self.act3(self.conv3(x))
#         # input 120x1x1, output 84
#         x = self.act4(self.fc1(self.flat(x)))
#         # input 84, output 10
#         x = self.fc2(x)
#         return x


#model = LeNet5()


optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    print("Working...")

torch.save(model, "modelf28.pth")

#import sys
#sys.exit()

p = Image.open("../test.png")
px = np.array(p)
#px = px[734:734+28, 474:474+28, 0]
px = px[472:472+28, 738:738+28, 0]
#plt.imshow(p)
plt.imshow(px)
plt.show()
#import sys
#sys.exit()

t = torch.from_numpy(px)
t2 = t.reshape(-1, 784).float() / 255.0

print(t2.shape, "T2 SHAPE")
y_pred = model(t2)
print(y_pred)
print(torch.max(y_pred, 1))
print(torch.argmax(y_pred, 1))

testloader = torch.utils.data.DataLoader(list(px, ), shuffle=False, batch_size=1)


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
#
# # Load MNIST data
# transform = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize((0,), (128,)),
# ])
# train = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
# test = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=100)
# testloader = torch.utils.data.DataLoader(test, shuffle=True, batch_size=100)
#
#
# class LeNet5(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
#         self.act1 = nn.Tanh()
#         self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
#
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
#         self.act2 = nn.Tanh()
#         self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
#
#         self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
#         self.act3 = nn.Tanh()
#
#         self.flat = nn.Flatten()
#         self.fc1 = nn.Linear(1 * 1 * 120, 84)
#         self.act4 = nn.Tanh()
#         self.fc2 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         # input 1x28x28, output 6x28x28
#         x = self.act1(self.conv1(x))
#         # input 6x28x28, output 6x14x14
#         x = self.pool1(x)
#         # input 6x14x14, output 16x10x10
#         x = self.act2(self.conv2(x))
#         # input 16x10x10, output 16x5x5
#         x = self.pool2(x)
#         # input 16x5x5, output 120x1x1
#         x = self.act3(self.conv3(x))
#         # input 120x1x1, output 84
#         x = self.act4(self.fc1(self.flat(x)))
#         # input 84, output 10
#         x = self.fc2(x)
#         return x
#
#
# model = LeNet5()
#
# optimizer = optim.Adam(model.parameters())
# loss_fn = nn.CrossEntropyLoss()
#
# n_epochs = 10
# for epoch in range(n_epochs):
#     model.train()
#     for X_batch, y_batch in trainloader:
#         y_pred = model(X_batch)
#         loss = loss_fn(y_pred, y_batch)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # Validation
#     model.eval()
#     acc = 0
#     count = 0
#     for X_batch, y_batch in testloader:
#         y_pred = model(X_batch)
#         acc += (torch.argmax(y_pred, 1) == y_batch).float().sum()
#         count += len(y_batch)
#     acc = acc / count
#     print("Epoch %d: model accuracy %.2f%%" % (epoch, acc * 100))