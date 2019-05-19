import numpy as np 

from keras.datasets import imdb
from keras.preprocessing  import sequence
from keras.models import Sequential
from keras.layers import Embedding, Dropout
from keras.layers import LSTM, Bidirectional, Dense

max_features = 20000  # 只要前20000万个常用单词

maxlen = 100  # 每个文本我们取它前100个单词

batch_size = 32  # 批量为32

# 1. 加载数据集
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)

print("训练集的数目:", len(x_train))
print("测试集的数目:", len(x_test))
# 输出：
# 训练集的数目: 25000
# 测试集的数目: 25000


# 2. 将每个文本pad成100个词  多余的单词直接扔掉
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_train, maxlen=maxlen)
print("训练数据的规格:", x_train.shape)
print("测试数据的规格:", x_test.shape)
# 输出:
# 训练数据的规格: (25000, 100)
# 测试数据的规格: (25000, 100)

# 3. 将标签的格式转一下
y_train = np.array(y_train)  
y_test = np.array(y_test)
print(y_train)    # 输出相当于一个列表，用0表示负样本 用1表示的正样本

# 4. 定义模型
model = Sequential()
# 注意词嵌入的时候 是输入的词的个数，以及词要嵌入到多少维向量，以及进行多少词的嵌入
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))   # 一层有64个LSTM的cell  其实相当于循环64次
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # 因为是二分类，我们这里采用sigmoid激活函数

# 5. 编译模型
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

# 6. 开始训练模型
print("开始训练模型")
model.fit(x_train, y_train, batch_size=batch_size, epochs=4, validation_data=[x_test, y_test])

# 7. 模型评估
loss, accuracy = model.evaluate(x_test, y_test)
print("损失:", loss)
print("准确率:", accuracy)

# 8. 保存模型
model.save('model.h5')


