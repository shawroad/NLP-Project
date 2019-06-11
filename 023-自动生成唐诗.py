"""

@file   : 023-自动生成唐诗.py

@author : xiaolu

@time   : 2019-06-11

"""
import numpy as np
from keras.layers import LSTM
from keras.layers import Dense, Embedding, Dropout
from keras.models import Sequential, load_model


def load_data(path):
    # 加载数据
    data = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        # print(len(lines))   # 可知有43030条数据
        # print(lines[0])   # 可以知道 每行代表一首诗
        for line in lines:
            line = line.replace('\n', '')
            data.append(line)
    return data


# 整理词表 并对数据进行预处理
def preprocess_file(path):
    files_content = ''
    puncs = [']', '[', '(', ')', '{', '}', ':', '《', '》']
    # 读取文本剔除以上的字符
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        lines = lines[:1000]
        for line in lines:
            # 每行的末尾加']'表示一首诗的结束
            for char in puncs:
                line = line.replace(char, '')
            files_content += line.strip() + ']'
    # print(files_content)
    words = list(files_content)

    # 统计每个字出现的次数
    word_count = {}
    for word in words:
        if word in word_count.keys():
            word_count[word] += 1
        else:
            word_count[word] = 1
    # 然后去除低频次
    erase = []
    for w, c in word_count.items():
        if c <= 2:
            erase.append(w)
    for k in erase:
        del word_count[k]

    del word_count[']']

    # 按照词频反向排字典
    wordPairs = sorted(word_count.items(), key=lambda x: -x[1])
    words, _ = zip(*wordPairs)

    # 整理词表
    word2id = {w: c+1 for c, w in enumerate(words)}
    id2word = {c: w for w, c in word2id.items()}
    return words, word2id, id2word, files_content


def generate_data(word2id, id2word, files_content):
    # 构造输入和输出
    # 1. 我们先将所有的输入转为数字
    word_list = list(files_content)
    data_num = []
    for w in word_list:
        id = word2id.get(w, 0)
        data_num.append(id)   # 一首诗的结束是0
    # print(data_num)

    # # 2. 开始构造输入   每次输入5个然后让其预测第六个字，依次这样走
    length = len(data_num)
    input_data = []
    target = []
    for i in range(length - 5):
        temp = data_num[i: i+5]
        input_data.append(temp)
        target.append(data_num[i+5])
    return np.array(input_data), target


def build_model(word_size, input_data, target):

    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(5, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(word_size, activation='softmax'))

    model.summary()   # 查看模型结构
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(input_data, target, epochs=1, batch_size=64)

    # 保存模型
    model.save("唐诗生成.h5")


if __name__ == '__main__':
    path = './data/poetry.txt'
    # data = load_data(path)
    # print(data[:3])
    words, word2id, id2word, files_content = preprocess_file(path)
    input_data, target = generate_data(word2id, id2word, files_content)
    # print(input_data[:3, :])
    # print(target[:3])
    word_size = len(words) + 1
    print(word_size)   # 2196去重后的字表大小

    input_data = np.reshape(input_data, (len(input_data), 5, 1))

    # 整理一下target  将其整理成one_hot编码
    target_ = np.zeros((len(target), word_size))
    for i, w in enumerate(target):
        target_[i][w] = 1

    # 建立模型
    build_model(word_size, input_data, target_)



    # # 测试
    # text = "路风两边走"
    # print(text)
    # # 1.先将文本转为数字
    # id_text = []
    # for w in list(text):
    #     id_text.append(word2id.get(w, 0))
    #
    # print(id_text)

    # 加载模型
    # model = load_model("唐诗生成.h5")
    # for i in range(10):
    #     id_text = np.array(id_text)
    #     id_text.reshape((1, 5))
    #     pred = model.predict(id_text)
    #     id = np.argmax(pred)
    #
    #     id_text = list(id_text[:, 1:]) + [id]
    #
    #     print(id2word.get(id))






