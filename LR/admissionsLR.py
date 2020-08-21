#!/bin/usr/python3
from dataLoader import data_loader
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

