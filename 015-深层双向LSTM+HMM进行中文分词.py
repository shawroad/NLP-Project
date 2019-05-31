"""

@file   : 015-深层双向LSTM+HMM进行中文分词.py

@author : xiaolu

@time1  : 2019-05-31

"""
# 将数据整理并进行字嵌入（Character Embedding）之后，使用Keras实现双向LSTM进行序列标注
from keras.layers import Input, Dense, Embedding, LSTM
from keras.layers import Dropout, TimeDistributed, Bidirectional
from keras.models import Model
from keras.utils import np_utils
import numpy as np
import re


# 读取字典   词表而已
vocab = open('./msr/msr_training_words.utf8', 'r', encoding='utf8').read().rstrip('\n').split('\n')  # 去除右边的换行，并按换行分割

vocab = list(''.join(vocab))   # 将所有词连接成一个大的字符串，然后转换为list  也就是说列表中每个元素为对应的字符
stat = {}
for v in vocab:
    stat[v] = stat.get(v, 0) + 1
stat = sorted(stat.items(), key=lambda x: x[1], reverse=True)
vocab = [s[0] for s in stat]   # 取出每一个字
# print(vocab)
# 5167 个字
# print(len(vocab))   # 看一下总的字个数

# 映射
char2id = {c: i + 1 for i, c in enumerate(vocab)}   # 字到id的映射
id2char = {i + 1: c for i, c in enumerate(vocab)}   # id到字的映射
# 这个tags就是我们最后的标签。
tags = {'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4}   # 进行的一些标注  x为填充的


# 定义模型所需要的一些参数
embedding_size = 128   # 字嵌入的长度
maxlen = 32  # 长于32则截断，短于32则填充0
hidden_size = 64
batch_size = 64
epochs = 50


# 定义一个读取并整理数据的函数
def load_data(path):
    data = open(path, 'r', encoding='utf8').read().rstrip('\n')
    # 按标点符号和换行符分隔
    data = re.split('[，。！？、\n]', data)  # 按 ,。！？、 \n 对数据进行分割 此时data中每个元素 只是文字和空格
    # 准备数据
    X_data = []
    y_data = []

    for sentence in data:
        sentence = sentence.split(' ')
        X = []
        y = []

        try:
            for s in sentence:   #
                s = s.strip()
                # 跳过空字符
                if len(s) == 0:
                    continue
                # s
                elif len(s) == 1:
                    # 一个字能够独立构成词的话  标记为s
                    X.append(char2id[s])
                    y.append(tags['s'])
                elif len(s) > 1:
                    # 如果多个字构成一个词， 第一个字标记为b  最后一个字标记为e  中间所有的字标记为m
                    # b
                    X.append(char2id[s[0]])
                    y.append(tags['b'])
                    # m
                    for i in range(1, len(s) - 1):
                        X.append(char2id[s[i]])
                        y.append(tags['m'])
                    # e
                    X.append(char2id[s[-1]])
                    y.append(tags['e'])

            # 统一长度    一个小句子的长度不能超过32,否则将其切断。只保留32个
            if len(X) > maxlen:
                X = X[:maxlen]
                y = y[:maxlen]
            else:
                for i in range(maxlen - len(X)):  # 如果长度不够的，我们进行填充，记得标记为x
                    X.append(0)
                    y.append(tags['x'])
        except:
            continue
        else:
            if len(X) > 0:
                X_data.append(X)   # [ [ [   每个词语中字转换为对应的id构成的列表     ], [], [], [] ...], [], [], [], []...]
                y_data.append(y)   # [ [ [每个词语中对应字的标注,有四种标记，构成的列表], [], [], []...], [], [], []...]

    X_data = np.array(X_data)
    y_data = np_utils.to_categorical(y_data, 5)   # 将y搞成one_hot编码

    return X_data, y_data


X_train, y_train = load_data('./msr/msr_training.utf8')    # 读取的这个文档是一个已经分好词的文本
X_test, y_test = load_data('./msr/msr_test_gold.utf8')
print('X_train size:', X_train.shape)    # (385152, 32)
print('y_train size:', y_train.shape)    # (385152, 32, 5)   # 标签转为one_hot编码  一句话中32个汉字对应六个标签 其中只有我们标记的那个为1
print('X_test size:', X_test.shape)    # (17917, 32)
print('y_test size:', y_test.shape)    # (17917, 32, 5)



# 定义模型
X = Input(shape=(maxlen,), dtype='int32')

# mask_zero的含义:我们在有些长度不够的字符串后面进行了零填充，这样在字嵌入的过程中，就会忽略这些零的存在
embedding = Embedding(input_dim=len(vocab) + 1, output_dim=embedding_size, input_length=maxlen, mask_zero=True)(X)

# 一般情况下，LSTM只会输出最后一层的输出，但是将return_sequences置为True  则会显示所有步的输出，所以他的输出和输入的长短就是一致的
# Bidirectional实现双向的LSTM  将正向的输出和反向的输出拼接到一起
blstm = Bidirectional(LSTM(hidden_size, return_sequences=True), merge_mode='concat')(embedding)  # hidden_size 相当于LSTM的输出维度
blstm = Dropout(0.6)(blstm)
blstm = Bidirectional(LSTM(hidden_size, return_sequences=True), merge_mode='concat')(blstm)
blstm = Dropout(0.6)(blstm)

# TimeDistributed作用: 上面的LSTM的return_sequences还是True  说明还是返回的一个长序列。
# 我们需要将长序列中的每一项都要进行softmax，这就是TimeDistributed的作用
output = TimeDistributed(Dense(5, activation='softmax'))(blstm)   # 此时的output形状 (batch_size, 32, 5)

model = Model(X, output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
model.save('msr_bilstm.h5')


# 查看模型在训练集和测试集上的分词正确率
print(model.evaluate(X_train, y_train, batch_size=batch_size))
print(model.evaluate(X_test, y_test, batch_size=batch_size))


# 定义维特比函数，使用动态规划算法获得最大概率路径
def viterbi(nodes):

    trans = {'be': 0.5, 'bm': 0.5, 'eb': 0.5, 'es': 0.5, 'me': 0.5, 'mm': 0.5, 'sb': 0.5, 'ss': 0.5}
    paths = {'b': nodes[0]['b'], 's': nodes[0]['s']}
    for l in range(1, len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1] + i in trans.keys():
                    nows[j + i] = paths_[j] + nodes[l][i] + trans[j[-1] + i]
            nows = sorted(nows.items(), key=lambda x: x[1], reverse=True)
            paths[nows[0][0]] = nows[0][1]

    paths = sorted(paths.items(), key=lambda x: x[1], reverse=True)
    return paths[0][0]


# 使用训练好的模型定义分词函数
def cut_words(data):
    data = re.split('[，。！？、\n]', data)   # 来一句话，我们先进行切分，因为我们的输入限制在32
    sens = []
    Xs = []
    for sentence in data:  # 按标点符号将很长的一句话转为几句很短的话
        sen = []
        X = []
        sentence = list(sentence)
        for s in sentence:
            s = s.strip()
            if not s == '' and s in char2id:
                sen.append(s)    # s添加到sen中
                X.append(char2id[s])   # 将s转为id 添加到X
        if len(X) > maxlen:   # 我们的模型长度有要求 所以这里得裁剪
            sen = sen[:maxlen]
            X = X[:maxlen]
        else:
            for i in range(maxlen - len(X)):   # 长度若不够  我们添加0 表示填充
                X.append(0)

        if len(sen) > 0:
            Xs.append(X)
            sens.append(sen)

    Xs = np.array(Xs)   # Xs的格式 [[子句1中字的id序列 并且长度为30], [子句2中字的id序列 并且长度为30], ....]
    ys = model.predict(Xs)   # 对每个字预测出五种概率，其中前四个概率是我们需要的，最后一个概率是对空的预测

    results = ''
    for i in range(ys.shape[0]):
        # 将每个概率与sbme对应构成字典 [{s:*, b:*, m:*, e:*}, {}, {}...]
        nodes = [dict(zip(['s', 'b', 'm', 'e'], d[:4])) for d in ys[i]]
        # [{'s': 1, 'b': 2, 'm': 3, 'e': 4}, {'s': 5, 'b': 6, 'm': 7, 'e': 8}]
        # 我测试的一个例子 里面的整数在这里应该是概率值  如列表中第一个字典 代表 “路” 这个字属于这四种标签的概率

        ts = viterbi(nodes)
        for x in range(len(sens[i])):
            if ts[x] in ['s', 'e']:
                results += sens[i][x] + '/'
            else:
                results += sens[i][x]

    return results[:-1]


# 调用分词函数
print(cut_words('中国共产党第十九次全国代表大会，是在全面建成小康社会决胜阶段、中国特色社会主义进入新时代的关键时期召开的一次十分重要的大会。'))
print(cut_words('把这本书推荐给，具有一定编程基础，希望了解数据分析、人工智能等知识领域，进一步提升个人技术能力的社会各界人士。'))
print(cut_words('结婚的和尚未结婚的。'))



