import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
import numpy as np

# Load MNIST data
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,), (128,)),
])
train = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
test = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=100)

fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)
for i in range(4):
    for j in range(4):
        ax[i][j].imshow(train.data[4 * i + j], cmap="gray")
plt.show()

print(train.data.shape, train.targets.shape)

X_train = train.data.reshape(-1, 784).float() / 255.0
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
    print(X_train.shape)

    # Save model
    torch.save(model.state_dict(), 'model.pth')


# Load model

model = Baseline()
model.load_state_dict(torch.load('model.pth'))
model.eval()

p = Image.open("../../test.png")
px = np.array(p)
px = px[472:472+28, 738:738+28, 0]
plt.imshow(px)
plt.show()

t = torch.from_numpy(px)
t2 = t.reshape(-1, 784).float() / 255.0
y_pred = model(t2)
print(y_pred)
print(torch.max(y_pred, 1))
print(torch.argmax(y_pred, 1))
