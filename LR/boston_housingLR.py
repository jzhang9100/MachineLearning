from LogisticRegression import LogisticRegression

import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()
print(x_train.shape)

#Feature Scaling
xtrain_trans = np.transpose(x_train)
for i, feature in enumerate(xtrain_trans):
    mean = np.mean(feature)
    std = np.std(feature)
    feature = (feature - mean)/std
    xtrain_trans[i] = feature
x_train = np.transpose(xtrain_trans)


xtest_trans = np.transpose(x_test)
for i, feature in enumerate(xtest_trans):
    mean = np.mean(feature)
    std = np.std(feature)
    feature = (feature-mean)/std
    xtest_trans[i] = feature
x_test = np.transpose(xtest_trans)

lr = LogisticRegression(x_train, y_train, x_test, y_test, 6000, 0.005)
lr.call(True)
loss, val_loss = lr.get_losses()

import matplotlib.pyplot as plt
plt.plot(loss, '--r', label='loss')
plt.plot(val_loss, '--b', label='val_loss')
plt.legend()
plt.savefig('boston_housingLR_loss.png')
