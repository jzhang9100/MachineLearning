#!/bin/usr/python3
#Jack Zhang
from model import ResNet50

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

res_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
res_optimizer = tf.keras.optimizers.Adam()

res_train_loss = tf.keras.metrics.Mean(name='train_loss')
res_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

res_test_loss = tf.keras.metrics.Mean(name='test_loss')
res_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def res_train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = resnet(images, training=True)
        l = res_loss(labels, predictions)
    gradients = tape.gradient(l, resnet.trainable_variables)
    res_optimizer.apply_gradients(zip(gradients, resnet.trainable_variables))

    res_train_loss(l)
    res_train_accuracy(labels, predictions)

@tf.function
def res_test_step(images, labels):
    predictions = resnet(images, training=False)
    t_l = res_loss(labels, predictions)

    res_test_loss(t_l)
    res_test_accuracy(labels, predictions)


EPOCHS = 50

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

    res_loss.append(train_loss.result())
    res_val_loss.append(res_test_loss.result())
    res_train_acc.append(res_train_accuracy.result())
    res_test_acc.append(res_test_accuracy.result())
print(model.summary())



#------------------------------------------------------------------------------------------#
#Vanilla without residual blocks
vanilla = VanillaModel()

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def vanilla_train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        l = loss(labels, predictions)
    gradients = tape.gradient(l, vanilla.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vanilla.trainable_variables))

    train_loss(l)
    train_accuracy(labels, predictions)

@tf.fucntion
def vanilla_test_step(images, labels):
    predictions = vanilla(images, training=False)
    t_l = loss(labels, predictions)

    test_loss(t_l)
    test_accuracy(labels, predictions)


#Vanilla Model
vanilla_loss = []
vanilla_val_loss = []
vanilla_train_acc = []
vanilla_test_acc = []
for epoch in range(EPOCHS):
    print("\nepoch {}/{}".format(epoch+1, EPOCHS))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    pbar = tf.keras.utils.Progbar(len(train))
    for i, d in enumerate(train):
        images, labels = d
        train_step(images, labels)
        pbar.update(i+1)

    for test_images, test_labels in test:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        test_loss.result(),
                        test_accuracy.result() * 100))

    vanilla_loss.append(train_loss.result())
    vanilla_val_loss.append(test_loss.result())
    vanilla_train_acc.append(train_accuracy.result())
    vanilla_test_acc.append(test_accuracy.result())
print(model.summary())


#Save Training Results
import matplotlib.pyplot as plt
plt.plot(vanilla_loss)
plt.plot(res_loss)
plt.savefig('train_loss.png')
plt.clf()

plt.plot(vanilla_val_loss)
plt.plot(res_val_loss)
plt.savefig('test_loss.png')
plt.clf()

plt.plot(vanilla_train_acc)
plt.plot(res_train_acc)
plt.savefig('train_acc.png')
plt.clf()

plt.plot(vanilla_test_acc)
plt.plot(res_test_acc)
plt.savefig('test_acc.png')

