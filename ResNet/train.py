#!/bin/usr/python3
#Jack Zhang
from model import ResNet50
from model import VanillaModel

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#----------------------------------------------------------------------------------------#

resnet = ResNet50()

res_loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
res_optimizer = tf.keras.optimizers.Adam()

res_train_loss = tf.keras.metrics.Mean(name='res_train_loss')
res_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='res_train_accuracy')

res_test_loss = tf.keras.metrics.Mean(name='res_test_loss')
res_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='res_test_accuracy')

@tf.function
def res_train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = resnet(images, training=True)
        r_l = res_loss_func(labels, predictions)
    gradients = tape.gradient(r_l, resnet.trainable_variables)
    res_optimizer.apply_gradients(zip(gradients, resnet.trainable_variables))

    res_train_loss(r_l)
    res_train_accuracy(labels, predictions)

@tf.function
def res_test_step(images, labels):
    predictions = resnet(images, training=False)
    r_t_l = res_loss_func(labels, predictions)

    res_test_loss(r_t_l)
    res_test_accuracy(labels, predictions)


EPOCHS = 2

#ResNet50 Train
res_loss = []
res_val_loss = []
res_train_acc = []
res_test_acc = []
for epoch in range(EPOCHS):
    print("\nepoch {}/{}".format(epoch+1,EPOCHS))
    res_train_loss.reset_states()
    res_train_accuracy.reset_states()
    res_test_loss.reset_states()
    res_test_accuracy.reset_states()

    pbar = tf.keras.utils.Progbar(len(train))
    for i, d in enumerate(train):
        images, labels = d
        res_train_step(images, labels)
        pbar.update(i+1)

    for test_images, test_labels in test:
        res_test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                        res_train_loss.result(),
                        res_train_accuracy.result() * 100,
                        res_test_loss.result(),
                        res_test_accuracy.result() * 100))

    res_loss.append(res_train_loss.result())
    res_val_loss.append(res_test_loss.result())
    res_train_acc.append(res_train_accuracy.result())
    res_test_acc.append(res_test_accuracy.result())
print(resnet.summary())



#------------------------------------------------------------------------------------------#
#Vanilla without residual blocks
vanilla = VanillaModel()

vanilla_loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
vanilla_optimizer = tf.keras.optimizers.Adam()

vanilla_train_loss = tf.keras.metrics.Mean(name='vanilla_train_loss')
vanilla_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='vanilla_train_accuracy')

vanilla_test_loss = tf.keras.metrics.Mean(name='vanilla_test_loss')
vanilla_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='vanilla_test_accuracy')

@tf.function
def vanilla_train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = vanilla(images, training=True)
        v_l = vanilla_loss_func(labels, predictions)
    gradients = tape.gradient(v_l, vanilla.trainable_variables)
    vanilla_optimizer.apply_gradients(zip(gradients, vanilla.trainable_variables))

    vanilla_train_loss(v_l)
    vanilla_train_accuracy(labels, predictions)

@tf.function
def vanilla_test_step(images, labels):
    predictions = vanilla(images, training=False)
    v_t_l = vanilla_loss_func(labels, predictions)

    vanilla_test_loss(v_t_l)
    vanilla_test_accuracy(labels, predictions)


#Vanilla Model
vanilla_loss = []
vanilla_val_loss = []
vanilla_train_acc = []
vanilla_test_acc = []
for epoch in range(EPOCHS):
    print("\nepoch {}/{}".format(epoch+1, EPOCHS))
    vanilla_train_loss.reset_states()
    vanilla_train_accuracy.reset_states()
    vanilla_test_loss.reset_states()
    vanilla_test_accuracy.reset_states()

    pbar = tf.keras.utils.Progbar(len(train))
    for i, d in enumerate(train):
        images, labels = d
        vanilla_train_step(images, labels)
        pbar.update(i+1)

    for test_images, test_labels in test:
        vanilla_test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                        vanilla_train_loss.result(),
                        vanilla_train_accuracy.result() * 100,
                        vanilla_test_loss.result(),
                        vanilla_test_accuracy.result() * 100))

    vanilla_loss.append(vanilla_train_loss.result())
    vanilla_val_loss.append(vanilla_test_loss.result())
    vanilla_train_acc.append(vanilla_train_accuracy.result())
    vanilla_test_acc.append(vanilla_test_accuracy.result())
print(vanilla.summary())


#Save Training Results
import matplotlib.pyplot as plt
plt.plot(vanilla_loss, '--r', label='vanilla_loss')
plt.plot(res_loss, '--b', label='res_loss')
plt.legend()
plt.savefig('train_loss.png')
plt.clf()

plt.plot(vanilla_val_loss, '--r', label='vanilla_test_loss')
plt.plot(res_val_loss, '--b', label='res_test_loss')
plt.savefig('test_loss.png')
plt.legend()
plt.clf()

plt.plot(vanilla_train_acc, '--r', label='vanilla_train_acc')
plt.plot(res_train_acc, '--b', label='res_train_acc')
plt.savefig('train_acc.png')
plt.legend()
plt.clf()

plt.plot(vanilla_test_acc, '--r', label='vanilla_test_acc')
plt.plot(res_test_acc, '--b', label='res_test_acc')
plt.legend()
plt.savefig('test_acc.png')

