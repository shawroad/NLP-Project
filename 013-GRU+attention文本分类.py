"""

@file   : 013-GRU+attention文本分类.py

@author : xiaolu

@time1  : 2019-05-28

"""
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense,merge,Input
from keras.layers import LSTM, Permute, Softmax, Lambda, Flatten, GRU
from keras import Model
import keras.backend as K
from keras.utils import to_categorical
import os

max_len = 200

# 加载语料
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 进行pad
train_data_pad = pad_sequences(train_data, padding="post", maxlen=max_len)
test_data_pad = pad_sequences(test_data, padding="post", maxlen=max_len)

# 标签进行one_hot
train_labels_input = to_categorical(train_labels)
test_labels_input = to_categorical(test_labels)

# 基于 特征词向量 整体的 attetion
# 最后权重得分是一个标量，将其特征词向量相乘 ，对特征词向量每个维度进行同样的缩放操作。
K.clear_session()
input_ = Input(shape=(max_len,))
words = Embedding(10000, 100, input_length=max_len)(input_)   # 词嵌入
sen = GRU(64, return_sequences=True)(words)   # [b_size, maxlen, 64]  一层64次

#attention
attention_pre = Dense(1, name='attention_vec')(sen)   # [b_size,maxlen,1]将每层输出直接sigmoid
attention_probs = Softmax()(attention_pre)  # [b_size,maxlen,1] 然后将输出的值进行softmax得出每个输出的权重
attention_mul = Lambda(lambda x: x[0]*x[1])([attention_probs, sen])  # 输出与权重相乘

output = Flatten()(attention_mul)    # 将所有的输出拉直 然后接几个Dense层
output = Dense(32, activation="relu")(output)
output = Dense(2, activation='softmax')(output)
model = Model(inputs=input_, outputs=output)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
model.summary()

# 训练模型
batch_size = 64
epochs = 2
model.fit(train_data_pad, train_labels_input, batch_size, epochs, validation_data=(test_data_pad, test_labels_input))

# 模型评估
loss, accuracy = model.evaluate(test_data_pad, test_labels_input)
print("损失:", loss)
print("准确率:", accuracy)

# 保存模型
model.save('FastTextClassify.h5')



# # 另外一种attention 这里的attention是向量
# # 最后 权重得分是一个向量,和特征词向量做element-wise的相乘，对特征词向量每个维度进行不同的缩放操作。
# K.clear_session()
# input_ = Input(shape=(max_len,))
# words = Embedding(10000, 100, input_length=max_len)(input_)
# sen = GRU(64, return_sequences=True)(words)  # [b_size,maxlen,64]
# 
# # attention
# attention_pre = Dense(64, name='attention_vec')(sen)   # [b_size,maxlen,64]
# attention_probs = Softmax()(attention_pre)  # [b_size,maxlen,64]
# attention_mul = Lambda(lambda x: x[0]*x[1])([attention_probs, sen])
# 
# output = Flatten()(attention_mul)
# output = Dense(32,activation="relu")(output)
# output = Dense(2, activation='softmax')(output)
# model = Model(inputs=input_, outputs=output)
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
# model.summary()
# 
# # 训练模型
# batch_size = 64
# epochs = 2
# model.fit(train_data_pad, train_labels_input, batch_size, epochs, validation_data=(test_data_pad, test_labels_input))
# 
# # 模型评估
# loss, accuracy = model.evaluate(test_data_pad, test_labels_input)
# print("损失:", loss)
# print("准确率:", accuracy)
# 
# # 保存模型
# model.save('FastTextClassify.h5')

