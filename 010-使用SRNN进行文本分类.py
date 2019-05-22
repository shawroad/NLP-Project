"""

@file   : 010-使用SRNN进行文本分类.py

@author : xiaolu

@time1  : 2019-05-22

"""
from keras.datasets import imdb
from keras.models import Model
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, GRU
from keras.layers import Input
from keras.layers import Dropout, TimeDistributed
import numpy as np
from keras.optimizers import Adam


max_feature = 10000
maxlen = 512   # 将文本截断成512个词
embedding_dims = 100   # 词嵌入100维
NUM_FILTERS = 50  # 相当于GRU单元循环50次
batch_size = 32


# 1.加载数据并预处理
print("加载数据。。。")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_feature)  # 只考虑前一万个常用词
# 我们现在把每篇文章pad成相同的长度
x_train = sequence.pad_sequences(x_train, maxlen)  # 截断成长度为512的向量
x_test = sequence.pad_sequences(x_test, maxlen)  # 截断成长度为512的向量


# 2.将数据进行分段
# 训练数据划分
x_train_split = []
for i in range(x_train.shape[0]):
    split1 = np.split(x_train[i], 8)  # 将每行文本分为8段
    a = []
    for j in range(8):
        s = np.split(split1[j], 8)  # 将每8段在进行分，分为8段
        a.append(s)
    x_train_split.append(a)  # 现在数据格式：[[这里面是一个文本，总共有64段], [], [], []...]
print(x_train_split)
# 测试数据划分
x_test_split = []
for i in range(x_test.shape[0]):
    split1 = np.split(x_test[i], 8)  # 将每行文本分为8段
    a = []
    for j in range(8):
        s = np.split(split1[j], 8)  # 将每8段在进行分，分为8段
        a.append(s)
    x_test_split.append(a)  # 现在数据格式：[[这里面是一个文本，总共有64段], [], [], []...]
print(x_test_split)


# 3.定义模型
input1 = Input(shape=(int(maxlen // 64),), dtype='int32')
embed = Embedding(max_feature, embedding_dims, input_length=maxlen)(input1)

gru1 = GRU(NUM_FILTERS, recurrent_activation='sigmoid', activation=None, return_sequences=False)(embed)
gru1 = Dropout(0.5)(gru1)
Encoder1 = Model(input1, gru1)

input2 = Input(shape=(8, int(300 // 64),), dtype='int32')
embed2 = TimeDistributed(Encoder1)(input2)
gru2 = GRU(NUM_FILTERS, recurrent_activation='sigmoid', activation=None, return_sequences=False)(embed2)
gru2 = Dropout(0.5)(gru2)
Encoder2 = Model(input2, gru2)

input3 = Input(shape=(8, 8, int(300 // 64)), dtype='int32')
embed3 = TimeDistributed(Encoder2)(input3)
gru3 = GRU(NUM_FILTERS, recurrent_activation='sigmoid', activation=None, return_sequences=False)(embed3)
gru3 = Dropout(0.5)(gru3)
preds = Dense(1, activation='sigmoid')(gru3)
model = Model(input3, preds)


# 4.定义Adam优化器 并进行模型编译
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


# 5.训练模型
model.fit(np.array(x_train_split), y_train, batch_size=batch_size, epochs=10)

# 6.评估模型
loss, accuracy = model.evaluate(np.array(x_test), y_test)
print("损失:", loss)
print("准确率:", accuracy)


# 7. 保存模型
model.save('model.h5')


