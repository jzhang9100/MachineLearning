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


'''
#Look at data
for sample in x_train:
    print(sample)
'''

#Hypothesis Function
def h(x,w):
    return np.dot(x, w)

#Cost Function
def J(T, w, Y, m):
    ret = 0
    for i in range(m):
        ret += (h(T[i], w) - Y[i])**2
    return ret * (1/m)


def stepSize(T, w, Y, j, m): #d/d0j
    ret = 0
    for i in range(m):
        ret += (h(T[i], w) - Y[i]) * T[i][j]

    return ret * (1/m)

def SGD(alpha, weights, data, labels, m):
    w = weights
    for j in range(len(w)):
        w[j] = w[j] - (alpha * stepSize(data, weights, labels, j, m))
    return w



N = 200
a = 0.005
m = len(x_train)
m_test = len(x_test)

def LogisticRegression(x_train, y_train, x_test, y_test, N, a):
    m = len(x_train)
    m_test = len(x_test)

print('Test Data Length: {}   Train Data Length: {}'.format(m_test, m))

weights = np.ones(len(x_train[0]))
train_loss = []
test_loss = []
for i in range(N):
    weights = SGD(a, weights, x_train, y_train, m)

    train_L = J(x_train, weights, y_train, m)
    test_L = J(x_test, weights, y_test, m_test)

    train_loss.append(train_L)
    test_loss.append(test_L)

    print('Iter #' + str(i) + ' Train Loss: ' + str(train_L) + ' Test Loss:' + str(test_L))


