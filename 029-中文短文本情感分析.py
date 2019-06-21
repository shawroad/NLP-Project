"""

@file   : 029-中文短文本情感分析

@author : xiaolu

@time   : 2019-06-21

"""
import re
import jieba
import pickle
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
from keras.layers import Embedding
import numpy as np
from keras.utils import to_categorical



def tokenizer(text):
    # 清洗 + 分词
    regex = re.compile(r'[^\u4e88-\u9fa5aA-Za-z0-9]')
    text = regex.sub('', text)     # 把汉字字母数字除外的全部字符扔掉
    return [word for word in jieba.cut(text)]


def get_stop_words():
    # 加载停用词表
    file_object = open('./data/stopwords.txt', encoding='utf8')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]   # 相当于去除'\n'
        line = line.strip()
        stop_words.append(line)
    return stop_words


def load_data(path, process):
    # 加载数据
    # process 表示数据要不要清洗 True为需要 False不需要直接加载
    if process == True:
        print("Data Loading...")
        labels = []
        data = []
        with open(path, 'r', encoding="utf8") as f:
            lines = f.readlines()
            for line in lines[1:]:    # 因为第一行是标题 我们直接扔掉
                line = line.replace('\n', '').split('\t')
                data.append(line[2])
                labels.append(line[1])

        out_label = open('labels.pkl', 'wb')
        pickle.dump(labels, out_label)
        # print(data[:3])
        # 将文本进行清洗
        temp_data = []
        for text in data:
            temp = tokenizer(text)
            temp_data.append(temp)

        print(temp_data[:3])
        # 过滤停用词
        stop_words = get_stop_words()   # 加载停用词
        for t in temp_data:
            for i in t:
                if i in stop_words:
                    t.remove(i)
        print(temp_data[:3])
        output = open('process_data.pkl', 'wb')
        pickle.dump(temp_data, output)

    else:
        f = open('process_data.pkl', 'rb')
        temp_data = pickle.load(f)
        f_l = open('labels.pkl', 'rb')
        labels = pickle.load(f_l)

    # 将词进行id的映射
    words = []
    for word in temp_data:
        words.extend(word)
    words = set(words)

    # 建立词和id之间的映射   0代表填充的所以从1开始
    word2id = {w: i+1 for i, w in enumerate(words)}
    # id2word = {i: w for w, i in word2id.items()}

    # 看一下平均长度
    length = 0
    i = 0
    for x in temp_data:
        length += len(x)
        i += 1
    average_length = int(length / i)

    # 将文本转为数字id 不够平均长度的直接pad成0
    id_data = [[word2id.get(w) for w in _] for _ in temp_data]
    for i in id_data:
        if len(i) < average_length:
            i.extend([0] * (average_length - len(i)))
        if len(i) > average_length:
            temp = i[:average_length]
            i.clear()
            i.extend(temp)

    return id_data, labels, word2id


def build_model(data, labels, word2id):
    # 用深层的双向LSTM
    vocab_size = len(word2id)
    model = Sequential()

    model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=11))
    model.add(Bidirectional(LSTM(64, dropout=0.1, activation='tanh', return_sequences=False)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(data, labels, batch_size=64, epochs=1, validation_split=0.2)

    # 保存模型
    model.save("情感分析.h5")


if __name__ == '__main__':
    path = './data/train.tsv'
    process = False
    data, labels, word2id = load_data(path, process)
    # 看一下前十条数据和标签
    # for i in range(10):
    #     print(data[i])
    #     print(labels[i])
    data, labels = np.array(data), np.array(labels)
    # 把标签整理一下
    labels = to_categorical(labels)

    # 建立模型并训练
    build_model(data, labels, word2id)
