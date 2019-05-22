"""

@file   : 011-文本生成之训练毛泽东语录.py

@author : xiaolu

@time1  : 2019-05-22

"""
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import ModelCheckpoint

with open('./data/chairmanmao.txt', 'r', encoding='gbk') as f:
    text = f.read()
    chars = sorted(list(set(text)))
    char_to_id = dict((c, i) for i, c in enumerate(chars))
    id_to_char = {i: w  for w, i in char_to_id.items()}

    n_chars = len(text)   # 整个语料的长度
    n_vocab = len(chars)  # 去重的字表

    print("语料的长度:", n_chars)   # 81586
    print("字表的长度:", n_vocab)   # 1696


seq_length = 100   # 打算用100预测一个
data_X = []
data_Y = []
for i in range(0, n_chars-seq_length):
    seq_in = text[i: i+seq_length]   # 前seq_length-1
    seq_out = text[i+seq_length]   # 第seq_length
    data_X.append([char_to_id[chars] for chars in seq_in])
    data_Y.append(char_to_id[seq_out])

# 统计训练文本条数
n_patterns = len(data_X)
print(n_patterns)    # 81486

X = np.reshape(data_X, (n_patterns, seq_length, 1))

# 将数据进行归一化
X = X / float(n_vocab)

# 进行标签的整理
y = to_categorical(data_Y)

# 定义模型
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam')

# 定义核查点
# 回调函数 Callbacks 是一组在训练的特定阶段被调用的函数集，你可以使用回调函数来观察训练过程中网络内部的状态和统计信息。
# 然后，在模型上调用 fit() 函数时，可以将 ModelCheckpoint 传递给训练过程。
# 训练深度学习模型时，Checkpoint 是模型的权重。ModelCheckpoint 回调类允许你定义检查模型权重的位置，文件应如何命名，
# 以及在什么情况下创建模型的 Checkpoint。
filepath = 'chinese_chairman_model.h5'
checkpoint = ModelCheckpoint(filepath,  monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=50, batch_size=128, callbacks=callbacks_list)


# 生成文本的阶段  给起始的种子进行预测
filename = "chinese_chairman_model.h5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# 读取一段其实文本，然后让其生成
with open('./data/sample_text.txt', 'r', encoding='gbk') as f:
    sample_text = f.read()
print(sample_text)
pattern = [char_to_id[c] for c in sample_text]
pattern = np.array(pattern)

for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    # 获得预测的当前字
    result = id_to_char[index]
    # pattern往后移动 预测下一个
    pattern.append(index)
    pattern = pattern[1:len(pattern)]






