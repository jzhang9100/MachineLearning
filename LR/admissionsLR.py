#!/bin/usr/python3
from dataLoader import data_loader
import numpy as np

d = data_loader(0.8)
d.graduation()
(x_train, y_train), (x_test, y_test) = d.load_data()
m = len(x_train)
print("Training Data Size: " + str(m))

#establish weights - 0 and baises b
w = np.ones(x_train.shape[0])
b = np.zeros(x_train.shape[0])


def h(x,w,b):
    #tanspose to get [9 x 400] * [400]i
    return x+b

t = h(x_train,w, b)
print(t.shape, t)
