#!/bin/usr/python3
from dataLoader import data_loader
import numpy as np

d = data_loader(0.8)
d.graduation()
(x_train, y_train), (x_test, y_test) = d.load_data()
m = len(x_train)
print("Training Data Size: " + str(m))


'''
#Look at data
for sample in x_train:
    print(sample)
'''
b = np.zeros(1)
w = np.ones(len(x_train[0]))

def h(x,w,b):
    return np.dot(x, w) + b

for x in x_train:
    print(h(x, w, b))
