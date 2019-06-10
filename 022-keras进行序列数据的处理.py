"""

@file   : 022-keras进行序列数据的处理.py

@author : xiaolu

@time1  : 2019-06-10

数据下载: https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip

"""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop

path = './data/jena_climate_2009_2016.csv'
with open(path, 'r') as f:
    lines = f.read().split('\n')

# 表头
header = lines[0].split(',')
data_length = len(lines[1:])
print("表头:", header)
print("数据条数:", data_length)   # 数据条数: 420551


# 解析数据
float_data = np.zeros((data_length, len(header) - 1))

for i, line in enumerate(lines[1:]):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values
print("数据规模:", float_data.shape)   # 数据规模:(420551, 14)


# # 绘制温度时间序列  温度序列是T列
# temp = float_data[:, 1]
# plt.plot(range(len(temp)), temp)
# plt.show()


# 数据标准化
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


# 生成时间序列样本及其目标的生成器
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    # data: 浮点数据
    # lookback: 输入数据应该包括过去的多少时间步
    # delay: 目标应该在未来多少个时间步之后
    # min_index和max_index: data数据索引 用来界定需要抽取那些时间步
    # shuffle: 打乱样本
    # batch_size: 批量数
    # step: 数据采样周期，我们设为6 表示每小时抽取一个数据点
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while True:
        if shuffle:
            # 从最小标到最大表中随机抽取batch_size条数据
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))  # 直接从i后面去batch条数据
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


lookback = 1440
step = 4
delay = 144
batch_size = 128

train_gen = generator(data=float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size
                      )
val_gen = generator(data=float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size
                    )
test_gen = generator(data=float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size
                     )

val_step = (300000 - 200001 - lookback) // batch_size
test_step = (len(float_data) - 300001 - lookback) // batch_size


# # 定义一个密集型模型
# model = Sequential()
# model.add(Flatten(input_shape=(lookback // step, float_data.shape[-1])))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1))
#
# model.compile(RMSprop(), loss='mae')
# history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20)


# 定义一个循环神经网络进行预测
model = Sequential()
model.add(GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(Dense(1))
model.compile(RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_step)



