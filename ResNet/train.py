#!/bin/usr/python3
#Jack Zhang
from model import char_model

import tensorflow as tf
import ipykernel

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

model = char_model()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        l = loss(labels, predictions)
    gradients = tape.gradient(l, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(l)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_l = loss(labels, predictions)

    test_loss(t_l)
    test_accuracy(labels, predictions)


EPOCHS = 5

for epoch in range(EPOCHS):
    print("\nepoch {}/{}".format(epoch+1,EPOCHS))
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
