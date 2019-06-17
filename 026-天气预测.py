"""

@file   : 018-天气预测.py

@author : xiaolu

@time   : 2019-06-17

"""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout
from keras.layers import Flatten, GRU, LSTM, TimeDistributed



def load_data(path):
    with open(path, 'r', encoding='utf8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            temp = line.replace('\n', '').split(',')
            data.append(temp)
    float_data = np.zeros((len(data), len(data[0]) - 1))
    for i, temp in enumerate(data[1:]):
        for j, d in enumerate(temp[1:]):
            float_data[i][j] = float(d)
    return float_data


def plot_image(float_data):
    # 画前九个属性的变化趋势
    for i in range(1, 10):
        plt.subplot("33{}".format(i))
        plt.plot(float_data[:, i])
    plt.show()


def generator_data(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    # 生成批数据

    if max_index is None:
        # 若没有指定max_index 这里将其赋值为最后索引减delay
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while True:
        if shuffle:
            # np.random.randint(a, b, size=c)  指的是在a到b索引之间随机取值  可能会取到重复值
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            # 数据若不打乱
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i+batch_size, max_index))
            i += len(rows)

        # 若是shuffle为True 从数据中随机选取batch个点， 这个点之前的lookback为观测数据，这个数据之后24小的数据为预测数据
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]   # 取出温度那一列
        yield samples, targets


def build_model1():
    # 密集型模型
    model = Sequential()
    model.add(Flatten(input_shape=(lookback // step, float_data.shape[-1])))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen, steps_per_epoch=500,  # steps_per_epoch指的是多少时间步算一个批次
                                  epochs=20, validation_data=val_gen,
                                  validation_steps=val_steps)

    # 画损失
    plt.figure(figsize=(20, 8))
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Vaildation loss')
    plt.legend()
    plt.show()


def build_model2():
    # GRU网络
    model = Sequential()
    model.add(GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, float_data.shape[-1])))
    model.add(Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40,
                                  validation_data=val_gen, validation_steps=val_steps)
    # 画损失
    plt.figure(figsize=(20, 8))
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Vaildation loss')
    plt.legend()
    plt.show()


def build_model3():

    # 堆叠多层GRU
    model = Sequential()
    model.add(GRU(12, dropout=0.1, recurrent_dropout=0.2,
                  return_sequences=True, input_shape=(None, float_data.shape[-1])))
    model.add(GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.3))
    model.add(Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40,
                                  validation_data=val_gen, validation_steps=val_steps)
    # 画损失
    plt.figure(figsize=(20, 8))
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Vaildation loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 初始的数据是每隔10分钟测一次
    path = './data/jena_climate_2009_2016.csv'
    float_data = load_data(path)

    # 画画几列数据的走向
    # plot_image(float_data)

    # 给定过去lookback多时间观测数据，预测未来delay步的数据  每隔step进行采样  知道10天预测11天
    lookback = 1440  # 10天
    step = 6  # 每个一小时采样依次
    delay = 144  # 目标是接下来的24小时天气
    batch_size = 128

    # 数据标准化
    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    float_data /= std

    # 生成批数据
    train_gen = generator_data(float_data, lookback=lookback,
                               delay=delay, min_index=0,
                               max_index=200000, shuffle=True,
                               step=step, batch_size=batch_size)
    val_gen = generator_data(float_data, lookback=lookback,
                             delay=delay, min_index=200001,
                             max_index=300000, step=step,
                             batch_size=batch_size)
    test_gen = generator_data(float_data, lookback=lookback,
                              delay=delay, min_index=300001,
                              max_index=None, step=step,
                              batch_size=batch_size)

    val_steps = (300000 - 200001 - lookback) // batch_size
    test_steps = (len(float_data) - 300001 - lookback) // batch_size

    # 定义密集连接模型
    # build_model1()
    # 定义GRU
    build_model2()
    # 定义DGRU
    # build_model3()


