"""

@file   : 030-卷积文本进行新闻分类.py

@author : xiaolu

@time   : 2019-06-24

"""
import re
import jieba
import numpy as np
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.layers import Embedding
from keras.layers import Input, Dropout, MaxPooling2D
from keras.layers import Reshape, Conv2D, Concatenate
from keras.models import Model



def load_data(path):
    with open(path, 'r', encoding='utf8') as f:
        labels = []
        data = []
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            label_temp = line[:2]
            labels.append(label_temp)
            data_temp = line[3:].strip()
            data.append(data_temp)

    return labels, data


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


def process_data(labels, data):
    # 先将标签和id进行映射
    label2id = {lab: i for i, lab in enumerate(list(set(labels)))}
    id2label = {i: lab for lab, i in label2id.items()}

    # 将我们的标签转为id
    label_num = []
    for lab in labels:
        label_num.append(label2id.get(lab))
    # 此时 label_num就是我们的标签列表

    word_data = []
    for d in data:
        word = tokenizer(d)
        word_data.append(word)
    # 此时word_data 就是分词后的数据

    # 过滤停用词
    stop_words = get_stop_words()  # 加载停用词
    for t in word_data:
        for i in t:
            if i in stop_words:
                t.remove(i)

    # 建立词表 并映射成id
    vocab = []
    for d in word_data:
        vocab.extend(d)
    vocab = list(set(vocab))
    vocab2id = {w: i+1 for i, w in enumerate(vocab)}
    # id2vocab = {i: w for w, i in vocab2id.items()}

    # 接下来，将我们文本转为id序列
    data_num = []
    for d in word_data:
        temp = [vocab2id.get(v) for v in d]
        data_num.append(temp)
    # 此时data_num是我们数据转为id序列数据

    # 看一下数据的平均长度
    i, length = 0, 0
    for d in data_num:
        i += 1
        length += len(d)
    average_len = length // i   # 平均长度
    # print(average_len)    # 507

    # pad成同样的长度
    for d in data_num:
        if len(d) < average_len:
            d.extend([0] * (average_len - len(d)))
        else:
            temp = d[:average_len]
            d.clear()
            d.extend(temp)

    return label_num, data_num, vocab2id, label2id, average_len


def get_model():
    seq_len = data.shape[1]
    num_filters = 64
    embedding_size = 128
    drop = 0.2

    filter_size = [2, 3, 4]
    inputs = Input(shape=(seq_len,))
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_size)(inputs)
    reshape = Reshape((seq_len, embedding_size, 1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_size[0], embedding_size), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)

    conv_1 = Conv2D(num_filters, kernel_size=(filter_size[1], embedding_size), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)

    conv_2 = Conv2D(num_filters, kernel_size=(filter_size[2], embedding_size), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)

    # 添加dropout层，防止过拟合
    conv_0 = Dropout(drop)(conv_0)
    conv_1 = Dropout(drop)(conv_1)
    conv_2 = Dropout(drop)(conv_2)

    # 添加池化
    maxpool_0 = MaxPooling2D(pool_size=(seq_len-filter_size[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPooling2D(pool_size=(seq_len-filter_size[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPooling2D(pool_size=(seq_len-filter_size[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

    # 将三种不同的卷积结果拼接到一块
    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])

    # 拉直
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    dense = Dense(32, activation='relu')(dropout)
    output = Dense(18, activation='softmax')(dense)

    model = Model(inputs=inputs, outputs=output)
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=data, y=label, epochs=10, batch_size=32, validation_split=0.1)
    model.save("卷积文本分类.h5")


if __name__ == '__main__':
    path = './data/news.txt'
    labels, data = load_data(path)
    cls_num = len(list(set(labels)))
    # print(len(labels))   # 1800
    # print(len(data))  # 1800
    # print(labels[:2])
    # print(data[:2])

    label_num, data_num, vocab2id, label2id, average_len = process_data(labels, data)

    vocab_size = len(vocab2id)
    # print(label_num)
    # for i in range(10):
    #     print(data_num[i])

    label = np.array(label_num)
    label = to_categorical(label)
    data = np.array(data_num)
    print(data.shape)    # (1800, 380)
    print(label.shape)   # (1800, 18)

    get_model()


