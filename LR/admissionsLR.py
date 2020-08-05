#!/bin/usr/python3
from dataLoader import data_loader

d = data_loader(0.8)
d.graduation()
(x_train, y_train), (x_test, y_test) = d.load_data()

