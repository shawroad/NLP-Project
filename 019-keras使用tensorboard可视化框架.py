import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential

max_features = 500
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len, name='embed'))
model.add(layers.Conv1D(16, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))   # 最大化池化
model.add(layers.Conv1D(16, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# 使用tensorboard回调函数来训练模型
callbacks = [
	keras.callbacks.TensorBoard(
		log_dir = 'model',   # 日志文件存在这个目录下
		histogram_freq=1,    # 每一轮之后记录激活直方图
		embeddings_freq=1,   # 每一轮之后记录嵌入数据
		embeddings_data = x_train.astype("float32")
		)
]

history = model.fit(x_train, y_train, epochs=1, batch_size=128, validation_split=0.2, callbacks=callbacks)

