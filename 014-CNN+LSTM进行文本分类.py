"""

@file   : 014-CNN+LSTM进行文本分类.py

@author : xiaolu

@time1  : 2019-05-30

"""
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dropout, Dense, Conv1D
from keras.layers import MaxPooling1D

max_feature = 10000   # 取出常用的前一万个词
max_len = 200  # 每个句子取出前100个词
embedding_size = 128  # 词嵌入的维度

# 加载数据
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_feature)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)


# 卷积
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

batch_size = 100
epochs = 2

# 定义模型
model = Sequential()
model.add(Embedding(max_feature, embedding_size, input_length=max_len))
model.add(Dropout(0.25))
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

# 保存模型
model.save('cnn+lstm.h5')

# 测试集的准确率
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

