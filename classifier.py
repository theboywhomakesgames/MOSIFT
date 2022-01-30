import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import numpy as np

import cv2

import json
import glob
import os
import random

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4096, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

data = []
labels = []
for filename in glob.glob('*.json'):
    path = os.path.join(os.getcwd(), filename)
    labels.append(filename)
    with open(path) as f:
        d = json.load(f)
        data.append(d)

data = np.array(data)
X = []
for action in data:
    a = []
    for feature in action:
        feature = np.array(feature)
        feature = np.reshape(feature, (64, 64))
        a.append(feature)
    X.append(a)

net = Net()
net.float()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.00001)

# train
for epoch in range(12):

    running_loss = 0.0
    for i in range(500):
        x = []
        l = []

        for j in range(32):
            label = random.randrange(0, len(X))
            f_index = random.randrange(0, len(X[label]) // 50) + len(X[label])//50
            inputs = X[label][f_index]
            img = np.array(inputs)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            inputs = np.expand_dims(inputs, 0)
            x.append(inputs)
            l.append(label)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        x = np.array(x)
        x = np.reshape(x, (32, 4096))
        x = torch.tensor(x, dtype=torch.float32)
        outputs = net(x)
        l = torch.tensor(l)
        loss = criterion(outputs, l)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# test the accuracy on the first chunk
accuracy = 0.0
corrects = 0
for i in range(100):
        label = random.randrange(0, len(X))
        f_index = random.randrange(0, len(X[label])//50) # first chunk
        inputs = X[label][f_index]
        inputs = np.expand_dims(inputs, 0)
        inputs = np.expand_dims(inputs, 0)
        inputs = torch.tensor(inputs, dtype=torch.float32)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        guess = torch.argmax(outputs)
        
        if guess == label:
            corrects += 1

accuracy = corrects / 100
print(accuracy)