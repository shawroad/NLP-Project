"""

@file   : 004-fastText实现文本分类.py

@author : xiaolu

@time1  : 2019-05-19

"""
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D


ngram_range = 2    # ngram_range = 2 will add bi-grams features
max_features = 20000    # 取前20000个常用词
batch_size = 32    # 每批32
embedding_dims = 50    # 词嵌入50维
epochs = 5    # 训练五个轮回


def create_ngram_set(input_list, ngram_value=2):
    # create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    # 输出:[(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    result = []
    for i in range(len(input_list) - ngram_value + 1):
        temp = tuple(input_list[i: i + ngram_value])
        result.append(temp)
    # print(set(result))
    return set(result)


def add_ngram(sequences, token_indice, ngram_range=2):
    # 若: sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    # 若: token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    # 调用此函数: add_ngram(sequences, token_indice, ngram_range=2)
    # 最后的输出: [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range+1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i: i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print("训练集的大小:", len(x_train))  # 训练集的大小: 25000
print("测试集的大小:", len(x_test))  # 测试集的大小: 25000

print("训练集文本的平均长度:", np.mean(list(map(len, x_train)), dtype=int))  # 训练集文本的平均长度: 238
print("测试集文本的平均长度:", np.mean(list(map(len, x_test)), dtype=int))  # 测试集文本的平均长度: 230

ngram_set = set()
if ngram_range > 1:
    for input_list in x_train:
        for i in range(2, ngram_range+1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)


    # 这里是将咱们上面做的那n-gram对应一个数字。  这里直接从max_features  因为前面的数字词已经用了
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}   # N-gram => 数字
    indice_token = {token_indice[k]: k for k in token_indice}    # 数字 => N-gram

    max_features = np.max(list(indice_token.keys())) + 1  # 这里特征数多了

    # 总结一下: 上面实现的就是特征的扩充，原本只是让词作为特征，
    # 这里有将连续的几个词作为特征。这样特征就扩充了好多

    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)

    # 咱们把每个文本的特征扩充了。 我们现在看看训练集和测试集的平均长度
    print("训练集文本的平均长度:", np.mean(list(map(len, x_train)), dtype=np.int))  # 训练集文本的平均长度: 476
    print("测试集文本的平均长度:", np.mean(list(map(len, x_test)), dtype=np.int))  # 测试集文本的平均长度: 428


# 为了统一往网络中喂  这里将每个文本都pad到同一个长度
maxlen = 400
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)   # 上面的平均长度时476  我们将所有统一填充到400
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)  # 上面的平均长度时428  我们也将所有统一填充到400

# 数据整理完毕 看一下
print("训练集的规格:", x_train.shape)
print("测试集的规格:", x_test.shape)


# 开始建立模型
model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
# 全局平均池化
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练走起
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# 模型评估
loss, accuracy = model.evaluate(x_test, y_test)
print("损失:", loss)
print("准确率:", accuracy)

# 保存模型
model.save('FastTextClassify.h5')









