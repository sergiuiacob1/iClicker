import math
import joblib
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
import os
import numpy as np


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
for x in os.listdir('./models'):
    if x.endswith('.json') == False:
        continue
    with open(f'models/{x}', 'r') as f:
        info = json.load(f)

    print (f'Treating {x}')

    # multiple line plot
    if info["prediction_type"] == "grid":
        lines = ['train_accuracy', 'test_accuracy',
                 'train_loss_categorical_crossentropy',
                 'test_loss_categorical_crossentropy',
                 ]
        colors = ["coral", "blue", "green", "red"]

        for line in lines:
            for index, val in enumerate(info[line]):
                info[line][index] = math.log(info[line][index])

        epochs = range(0, len(info['train_accuracy']))
    else:
        lines = ['train_loss_mean_squared_error',
                 'test_loss_mean_squared_error']
        colors = ["green", "red"]

        for line in lines:
            for index, val in enumerate(info[line]):
                info[line][index] = math.log(info[line][index])

        epochs = range(0, len(info['train_loss_mean_squared_error']))
    for i in range(0, len(lines)):
        ax = sns.lineplot(
            x=epochs, y=lines[i], data=info,  color=colors[i], label=lines[i])
    # plt.show(sns)
    figname = x.replace('.json', '.png')
    plt.xlabel('epochs')
    plt.savefig(f'report/images/graphs/{figname}')
    # plt.savefig(f'/Users/sergiuiacob/Desktop/models/{figname}')
    plt.clf()
