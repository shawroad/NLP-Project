"""

@file   : 016-LSTM+Attention通过人名预测性别.py

@author : xiaolu

@time1  : 2019-06-03

"""
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Embedding, Input, Softmax, Lambda
from keras.layers import LSTM, Dropout, Dense, Flatten
import keras.backend as K



def load_data(path):
    data = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            data.append(line)
    return data


def process_data(data_male, data_female):
    # 1.剔除一些即是男名有时女名的名字
    for name in data_female:
        if name in data_male:
            data_female.remove(name)
            data_male.remove(name)
    print("男性名字的个数:", len(data_male))   # 2629
    print("女性名字的个数:", len(data_female))   # 4687
    # 样本不太均衡  我们可以去除男性的前三千个名字
    data_male = data_male[:3000]

    # 2.接下来我们看男性和女性中 那个名字最长
    maxlen = 0
    for name in data_male:
        if len(name) > maxlen:
            maxlen = len(name)
    for name in data_female:
        if len(name) > maxlen:
            maxlen = len(name)

    # 3. 进行字符id的映射
    names = []
    names.extend(data_female)
    names.extend(data_male)
    str = ''.join(names)
    str = set(list(str.lower()))
    char2id = {c: i+1 for i, c in enumerate(str)}  # 之所以空下1 就是为那些不存在的字符准备的
    char2id['UNK'] = 0

    id2char = {i: c for c, i in char2id.items()}

    # 4.进行填充 将其所有名字填充为maxlen长  并转为数字
    data_male_id = []
    temp = []
    for name in data_male:
        name = name.lower()
        for c in name:
            temp.append(char2id.get(c, 0))
        data_male_id.append(temp)
        temp = []

    data_female_id = []
    temp = []
    for name in data_female:
        name = name.lower()
        for c in name:
            temp.append(char2id.get(c, 0))
        data_female_id.append(temp)
        temp = []

    print("前两个男性名:", data_male_id[:2])
    print("前两个女性名:", data_female_id[:2])

    # 进行pad
    data_male_id = pad_sequences(data_male_id, padding='post', maxlen=maxlen, value=0)
    data_female_id = pad_sequences(data_female_id, padding='post', maxlen=maxlen, value=0)

    print("前两个男性名:", data_male_id[:2])
    print("前两个女性名:", data_female_id[:2])

    return data_male_id, data_female_id, char2id, id2char, maxlen



def build_model(data, label):
    embedding_dim = 200
    size_char = len(char2id)
    n_epoch = 10
    batch_size = 50

    K.clear_session()
    input_ = Input(shape=(maxlen,))
    words = Embedding(size_char, embedding_dim, input_length=maxlen)(input_)  # 词嵌入
    sen = LSTM(64, return_sequences=True)(words)  # [b_size, maxlen, 64]

    # attention  这里是将每层的输出连接Dense 最后接softmax  attention最后是个概率标量
    attention_pre = Dense(1, name='attention_vec')(sen)  # [b_size,maxlen,1]将每层输出直接sigmoid
    attention_probs = Softmax()(attention_pre)  # [b_size,maxlen,1] 然后将输出的值进行softmax得出每个输出的权重
    attention_mul = Lambda(lambda x: x[0] * x[1])([attention_probs, sen])  # 输出与权重相乘

    output = Flatten()(attention_mul)  # 将所有的输出拉直 然后接几个Dense层
    output = Dense(32, activation="relu")(output)
    output = Dense(1, activation='sigmoid')(output)
    model = Model(inputs=input_, outputs=output)

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(data, label, batch_size=batch_size, epochs=n_epoch)
    model.summary()

    # # 这个模型稍微简单些
    # model = Sequential()
    # model.add(Embedding(input_dim=size_char, output_dim=embedding_dim, input_length=maxlen))
    # model.add(LSTM(128, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(128, return_sequences=False))
    # model.add(Dropout(0.2))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # model.fit(data, label, batch_size=batch_size, epochs=n_epoch)
    # model.summary()

    # 保存模型
    model.save_weights('my_model.h5')
    # 预测
    score = model.evaluate(data, label, batch_size=16)
    print("准确率:", score)


if __name__ == "__main__":
    male_path = './data/male.txt'
    female_path = './data/female.txt'
    data_male = load_data(male_path)
    data_female = load_data(female_path)
    print("前十个男性名:", data_male[:10])
    print("前十个女性名:", data_female[:10])
    print("男性名字的个数:", len(data_male))   # 2943
    print("女性名字的个数:", len(data_female))  # 5001

    data_male_id, data_female_id, char2id, id2char, maxlen = process_data(data_male, data_female)
    # 制作标签
    data_male_label = np.array([1] * len(data_male_id)).reshape((-1, 1))
    data_female_label = np.array([0] * len(data_female_id)).reshape((-1, 1))
    # 数据格式调整一下
    data_male_id = np.array(data_male_id)
    data_female_id = np.array(data_female_id)
    data = np.r_[data_male_id, data_female_id]
    label = np.r_[data_male_label, data_female_label]

    build_model(data, label)




