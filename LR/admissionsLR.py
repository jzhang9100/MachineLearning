#!/bin/usr/python3
from dataLoader import data_loader
from LogisticRegression import LogisticRegression
import numpy as np

d = data_loader(0.8)
d.graduation()
(x_train, y_train), (x_test, y_test) = d.load_data()


#feature scaling
xtrain_T = np.transpose(x_train)
for i, feature in enumerate(xtrain_T):
    mean = np.mean(feature)
    std = np.std(feature)
    feature = (feature-mean)/std
    xtrain_T[i] = feature
x_train = np.transpose(xtrain_T)

xtest_T = np.transpose(x_test)
for i, feature in enumerate(xtest_T):
    mean = np.mean(feature)
    std = np.std(feature)
    feature = (feature-mean)/std
    xtest_T[i] = feature
x_test = np.transpose(xtest_T)

lr = LogisticRegression(x_train, y_train, x_test, y_test, 5000, 0.005)
lr.call()
loss, val_loss = lr.get_losses()

import matplotlib.pyplot as plt
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.savefig('admissions_data_LR_loss.png')
