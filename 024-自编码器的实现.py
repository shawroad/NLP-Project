"""

@file   : 024-自编码器的实现.py

@author : xiaolu

@time   : 2019-06-13

"""
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 定义网络结构
input_img = Input(shape=(28, 28, 1))

# 编码过程
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# 解码过程
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 得到编码层的输出
encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('conv2d_4').output)


# 可视化训练结果， 我们打开终端， 使用tensorboard
# tensorboard --logdir=/tmp/autoencoder # 注意这里是打开一个终端， 在终端里运行
# 训练模型， 并且在callbacks中使用tensorBoard实例， 写入训练日志 http://0.0.0.0:6006
# from keras.callbacks import TensorBoard
# autoencoder.fit(x_train, x_train,
#                 epochs=50,
#                 batch_size=128,
#                 shuffle=True,
#                 validation_data=(x_test, x_test),
#                 callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


autoencoder.fit(x_train, x_train, epochs=1, batch_size=64, shuffle=True, validation_data=(x_test, x_test))

# 保存模型
autoencoder.save("自编码器.h5")

# 重建图片
import matplotlib.pyplot as plt
decoded_imgs = autoencoder.predict(x_test)
encoded_imgs = encoder_model.predict(x_test)
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    k = i + 1
    # 画原始图片
    ax = plt.subplot(2, n, k)
    plt.imshow(x_test[k].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    # 画重建图片
    ax = plt.subplot(2, n, k + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# 编码得到的特征
n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    k = i + 1
    ax = plt.subplot(1, n, k)
    plt.imshow(encoded[k].reshape(4, 4 * 8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()