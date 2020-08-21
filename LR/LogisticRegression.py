#!/bin/usr/python3
import numpy as np

class LogisticRegression:
    def __init__(self, x_train, y_train, x_test, y_test, N, a):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.N = N
        self.a = a
        self.M = len(x_train)
        self.m = len(x_test)
        self.weights = np.ones(len(x_train[0]))
        self.train_loss = []
        self.test_loss = []


    def call(self):
        for i in range(self.N):
            self.weights = self.SGD(self.a, self.weights, self.x_train, self.y_train, self.M)

            train_L = self.J(self.x_train, self.weights, self.y_train, self.M)
            test_L = self.J(self.x_test, self.weights, self.y_test, self.m)

            self.train_loss.append(train_L)
            self.test_loss.append(test_L)

            print('Iter #' + str(i) + ' Train Loss: ' + str(train_L) + ' Test Loss:' + str(test_L))

    def get_losses(self):
        return self.train_loss, self.test_loss


    #Hypothesis Function
    def h(self, x, w):
        return np.dot(x, w)

    #Cost Function
    def J(self, T, w, Y, m):
        ret = 0
        for i in range(m):
            ret += (self.h(T[i], w) - Y[i])**2
        return ret * (1/m)


    def stepSize(self, T, w, Y, j, m): #d/d0j
        ret = 0
        for i in range(m):
            ret += (self.h(T[i], w) - Y[i]) * T[i][j]

        return ret * (1/m)

    def SGD(self, alpha, weights, data, labels, m):
        w = weights
        for j in range(len(w)):
            w[j] = w[j] - (alpha * self.stepSize(data, weights, labels, j, m))
        return w
