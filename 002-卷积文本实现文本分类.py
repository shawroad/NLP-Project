from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D

# 我们这里使用的是imdb数据集

# 定义一些参数
max_features = 5000  # 取常用的5000个词
maxlen = 400   # 每个文本取前400个词

batch_size = 32
embedding_dims = 50 # 将词嵌入成50维向量
filters = 250  # 卷积核为250个
kernel_size = 3  # 一维卷积核的大小 长度为3
hidden_dim = 250

epochs = 2  # 训练两个轮回

# 1. 加载数据集
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print("训练集的大小:", len(x_train))
print("测试集的大小:", len(x_test))


# 2. 将每篇文章pad到长度为400
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print("训练文本的规格:", x_train.shape)
print("测试文本的规格:", x_test.shape)


# 3.建立模型

model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(Dropout(0.2))
# 卷积核的大小[kernel_size, embedding_dims] 只是纵向卷积  不会横向移动
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))   
# 进行最大化池化
model.add(GlobalMaxPooling1D())   # 这样确保每篇文章最后压缩的长度等于卷积核的个数
# 一番卷积过后  我们两个全连接层
model.add(Dense(hidden_dim))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# 4. 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# 5. 训练模型
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# 6. 模型评估
loss, accuracy = model.evaluate(x_test, y_test)
print("损失:", loss)
print("准确率:", accuracy)

# 7. 保存模型
model.save('model.h5')





