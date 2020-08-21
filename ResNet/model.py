#!/bin/usr/python3
#Jack Zhang

import tensorflow as tf

from tensorflow.keras.layers import Activation, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Add, Dropout
from tensorflow.keras import Model

class VanillaModel(Model):
    def __init__(self):
        super(VanillaModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu', input_shape=(32,32,32,3))
        self.pool1 = MaxPooling2D(pool_size=(3,3))

        self.nn = []
        i = 64
        while i <= 256:
            self.nn.append(self.non_res_block(i))
            self.nn.append(self.non_res_block(i))
            self.nn.append(self.non_res_block(i))
            i*=2

        self.bn1 = BatchNormalization()
        self.pool2 = MaxPooling2D(pool_size=(7,7))
        self.flat = Flatten()
        self.fc = Dense(1024, activation='relu')
        self.drop = Dropout(0.5)
        self.out = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        for block in self.nn:
            for layer in block:
                x = layer(x)

        x = self.pool2(x)
        x = self.flat(x)
        x = self.fc(x)
        x = self.drop(x)

        return self.out(x)




    #Returns a non Residual block
    def non_res_block(self, filters):
        #a
        c1 = Conv2D(filters, kernel_size=(1,1), padding='valid', activation=None)
        bn1 = BatchNormalization()
        a1 = Activation('relu')
        #b
        c2 = Conv2D(filters, kernel_size=(3,3), padding='same', activation=None)
        bn2 = BatchNormalization()
        a2 = Activation('relu')

        #c
        c3 = Conv2D(4*filters, kernel_size=(1,1), padding='valid',activation=None)
        bn3 = BatchNormalization()
        a3 = Activation('relu')

        return [c1, bn1, a1, c2, bn2, a2, c3, bn3, a3]

class ResNet50(Model):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu', input_shape=(32,32,32,3))
        self.pool1 = MaxPooling2D(pool_size=(3,3))

        self.nn = []
        i = 64
        while i <= 256:
            self.nn.append(self.res_net_block(i))
            self.nn.append(self.non_res_block(i))
            self.nn.append(self.non_res_block(i))
            i = i*2

        self.add = Add()
        self.relu = Activation('relu')
        self.bn1 = BatchNormalization()

        self.pool2 = MaxPooling2D(pool_size=(7,7))
        self.flat = Flatten()
        self.fc = Dense(1024, activation='relu')
        self.drop = Dropout(0.5)
        self.out = Dense(10)


    def call(self, x):
        #print(x.shape)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        for block in self.nn:
            #Residual Block
            if len(block) == 10:
                x_ = x
                for i in range(len(block)-2):
                    x = block[i](x)

                x_ = block[len(block)-2](x_)
                x_ = block[len(block)-1](x_)

                x = self.add([x, x_])
                x = self.relu(x)

            #Non Res Block
            else:
                x_ = x
                for layer in block:
                    x = layer(x)

                x = self.add([x, x_])
                x = self.relu(x)

        x = self.pool2(x)
        x = self.flat(x)
        x = self.fc(x)
        x = self.drop(x)

        return self.out(x)

    #Returns a Residual block
    def res_net_block(self, filters):
        #a
        c1 = Conv2D(filters, kernel_size=(1,1), padding='valid', activation=None)
        a1 = Activation('relu')
        bn1 = BatchNormalization()

        #b
        c2 = Conv2D(filters, kernel_size=(3,3), padding='same', activation=None)
        bn2 = BatchNormalization()
        a2 = Activation('relu')

        #c
        c3 = Conv2D(4*filters, kernel_size=(1,1), padding='valid',activation=None)
        bn3 = BatchNormalization()

        #Residual
        c4 = Conv2D(4*filters, kernel_size=(1,1), padding='valid',activation=None)
        bn4 = BatchNormalization()

        return [c1, a1, bn1, c2, a2, bn2, c3, bn3, c4, bn4]

    #Returns a non Residual block
    def non_res_block(self, filters):
        #a
        c1 = Conv2D(filters, kernel_size=(1,1), padding='valid', activation=None)
        bn1 = BatchNormalization()
        a1 = Activation('relu')

        #b
        c2 = Conv2D(filters, kernel_size=(3,3), padding='same', activation=None)
        bn2 = BatchNormalization()
        a2 = Activation('relu')

        #c
        c3 = Conv2D(4*filters, kernel_size=(1,1), padding='valid',activation=None)
        bn3 = BatchNormalization()

        return [c1, a1, bn1, c2, a2, bn2, c3, bn3]
