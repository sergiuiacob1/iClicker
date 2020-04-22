import joblib
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
import os
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


# convert data to torch.FloatTensor
transform = transforms.ToTensor()
# load all of the heatmaps
h_list = os.listdir('./crowdpupil/heatmaps')
all_data = []
for h in h_list:
    img = cv2.imread('./crowdpupil/heatmaps/' + h, cv2.IMREAD_GRAYSCALE)
    # resize the image to a fixed dimension
    img = cv2.resize(img, (1000, 776), interpolation=cv2.INTER_AREA)
    img = img.reshape(1, 1000, 776)
    img = img/255
    img = torch.from_numpy(img).float()
    all_data.append(img)

print(len(all_data))

# split train test
random.shuffle(all_data)
split_size = 0.8
train_size = int(len(all_data) * split_size)
train_data = all_data[:train_size]
test_data = all_data[train_size:]

print(len(train_data))

# Create training and test dataloaders

num_workers = 0
# how many samples per batch to load
batch_size = 20

# prepare data loaders
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, num_workers=num_workers)


# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        # a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation

        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))

        return x


# initialize the NN
model = ConvAutoencoder()
print(model)

# specify loss function
criterion = nn.MSELoss()

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# number of epochs to train the model
n_epochs = 30

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0

    ###################
    # train the model #
    ###################
    for data in train_loader:
        # _ stands in for labels, here
        # no need to flatten images
        images = data
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)
        # calculate the loss
        loss = criterion(outputs, images)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*images.size(0)

    # print avg training statistics
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch,
        train_loss
    ))

# which = 'eye_strips_regression.pkl'
# data = joblib.load(f'./train_data/{which}')

# print(len(data[0]))
# print(data[1][1])
# print(data[0][0].shape)
# for i in range(0, len(data[1][1])):
#     summm = sum([1 for x in data[1] if x[i] == 1])
#     print(f'Class {i} has {summm} instances')

# n = len(data[0])
# for i in range(0, 1):
#     r = random.randint(0, n - 1)
#     img = data[0][r]
#     print(data[1][r])
#     cv2.imshow(f'{i}', img)
#     cv2.waitKey(0)

# cv2.destroyAllWindows()


# PLOTS
# for x in os.listdir('./models'):
#     if x.endswith('.json') == False:
#         continue
#     with open(f'models/{x}', 'r') as f:
#         info = json.load(f)

#     # multiple line plot
#     if info["prediction_type"] == "grid":
#         lines = ['train_accuracy', 'test_accuracy',
#                  'train_loss_categorical_crossentropy', 'test_loss_categorical_crossentropy']
#         colors = ["coral", "blue", "green", "red"]
#         epochs = range(0, len(info['train_accuracy']))
#     else:
#         lines = ['train_loss_mean_squared_error',
#                  'test_loss_mean_squared_error']
#         colors = ["green", "red"]
#         epochs = range(0, len(info['train_loss_mean_squared_error']))
#     for i in range(0, len(lines)):
#         ax = sns.lineplot(
#             x=epochs, y=lines[i], data=info,  color=colors[i], label=lines[i])
#     # plt.show(sns)
#     figname = x.replace('.json', '.png')
#     plt.xlabel('epochs')
#     plt.savefig(f'report/images/graphs/{figname}')
#     plt.clf()
