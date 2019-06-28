"""

@file   : 031-自编码器进行图像去燥.py

@author : xiaolu

@time   : 2019-06-27

"""
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers import *

(x_train, _), (x_test, _) = mnist.load_data()

# 标准化
x_train = x_train.reshape(x_train.shape + (1,)) / 255.
x_test = x_test.reshape(x_test.shape + (1,)) / 255.
print(x_train.shape)   # (60000, 28, 28, 1)
# plt.imshow(x_train[0])
# plt.show()

# 加白噪声
noise_factor = 0.5   # 噪声因子
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)   # 裁掉那些超出0和1的值
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
# print(x_test_noisy.shape)


# 输出9对图
# plt.gray()
# for i in range(1, 10):
#     plt.subplot(2, 9, i)
#     plt.imshow(x_train[i])
#     plt.subplot(2, 9, i+9)
#     plt.imshow(x_train_noisy[i])
# plt.show()


# 定义模型
input_img = Input(shape=(28, 28, 1, ))

# 实现encoder部分，由两个 3x3x32 的卷积和两个 2x2 的最大池化组成
x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_img)  # 28x28x32
x = MaxPool2D((2, 2), padding='same')(x)    # 14x14x32
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)   # 14x14x32
encoded = MaxPool2D((2, 2), padding='same')(x)   # 7x7x32


# 实现decoder部分, 由两个 3x3x32 的卷积和两个 2x2 的上采样组成
x = Conv2D(32, (3, 3), padding='same', activation='relu')(encoded)  # 7x7x32
x = UpSampling2D((2, 2))(x)   # 14x14x32
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)  # 14x14x32
x = UpSampling2D((2, 2))(x)  # 28x28x32
decoded = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)  # 28x28x1

# 将输入和输出连接，构造自编码器并compile
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

autoencoder.fit(x_train_noisy, x_train, epochs=1, batch_size=64)

autoencoder.save("图像去燥.h5")
