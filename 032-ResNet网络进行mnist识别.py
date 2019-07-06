"""

@file   : 032-ResNet网络进行mnist识别.py

@author : xiaolu

@time   : 2019-07-05

"""
# ResNet50标准层首层224*224*3
import keras.layers as KL
from keras.models import Model
import keras.backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from keras.datasets import mnist
from  keras.utils.vis_utils import plot_model

# 创建简单的ResNet
def building_block(filters, block):
    # 判断block1和2
    if block != 0:   # 如果不等于0 那么使用 stride=1
        stride = 1
    else:
        stride = 2   # 两倍下采样

    def f(x):
        # 主通路结构
        y = KL.Conv2D(filters=filters, kernel_size=(1, 1), strides=stride)(x)
        y = KL.BatchNormalization()(y)
        y = KL.Activation('relu')(y)

        y = KL.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(y)
        y = KL.BatchNormalization()(y)
        y = KL.Activation('relu')(y)
        # 主通路输出
        y = KL.Conv2D(filters=4*filters, kernel_size=(1, 1))(y)
        y = KL.BatchNormalization()(y)

        # 判断不同的block 设定不同的shortcut支路参数
        if block == 0:
            shortcut = KL.Conv2D(filters=4*filters, kernel_size=(1, 1), strides=stride)(x)
            shortcut = KL.BatchNormalization()(shortcut)
        else:     # 如果不等于0 那就是block2  那么就直接接input的tensor
            shortcut = x

        # 主路和shortcut相加
        y = KL.Add()([y, shortcut])
        y = KL.Activation('relu')(y)
        return y

    return f


def ResNet_Extractor(x_train, y_train, x_test, y_test):

    input = KL.Input(shape=(28, 28, 1))
    x = KL.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)

    filters = 64
    block = [2, 2]  # 用两层下采样的卷积
    for i, block_num in enumerate(block):
        for block_id in range(block_num):
            x = building_block(filters=filters, block=block_id)(x)
        filters *= 2

    x = KL.AveragePooling2D(pool_size=(2, 2))(x)
    x = KL.Flatten()(x)
    x = KL.Dense(10, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)

    model.summary()
    # 将网络结构打印出来
    # plot_model(model, to_file='res_mnist.png', show_shapes=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=1, batch_size=128, validation_data=(x_test, y_test))

    # 保存模型
    model.save('res_mnist.h5')


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    ResNet_Extractor(x_train, y_train, x_test, y_test)







