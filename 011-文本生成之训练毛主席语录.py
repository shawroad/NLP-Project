"""

@file   : 011-文本生成之训练毛主席语录.py

@author : xiaolu

@time1  : 2019-05-22

"""
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils


filename = "shiji_sample.txt"

with open('./data/sample_text.txt', 'r', encoding='gbk') as f:
    text = f.read()


#
# filename = "shiji_sample.txt"
# raw_text = unicode(open(filename).read(), "utf-8")
#
# # create mapping of unique chars to integers, and a reverse mapping
# chars = sorted(list(set(raw_text)))
# char_to_int = dict((c, i) for i, c in enumerate(chars))
# int_to_char = dict((i, c) for i, c in enumerate(chars))
# # summarize the loaded data
# n_chars = len(raw_text)
# n_vocab = len(chars)
# print("Total Characters: ", n_chars)
# print("Total Vocab: ", n_vocab)
# # prepare the dataset of input to output pairs encoded as integers
# seq_length = 20
# dataX = []
# dataY = []
# for i in range(0, n_chars - seq_length, 1):
# 	seq_in = raw_text[i:i + seq_length]
# 	seq_out = raw_text[i + seq_length]
# 	dataX.append([char_to_int[char] for char in seq_in])
# 	dataY.append(char_to_int[seq_out])
# n_patterns = len(dataX)
# print("Total Patterns: ", n_patterns)
# # reshape X to be [samples, time steps, features]
# X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# # normalize
# X = X / float(n_vocab)
# # one hot encode the output variable
# y = np_utils.to_categorical(dataY)
# # define the LSTM model
# model = Sequential()
# model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(256))
# model.add(Dropout(0.2))
# model.add(Dense(y.shape[1], activation='softmax'))
# # load the network weights
# filename = "chinese_shiji_lstm_model_20161123"
# model.load_weights(filename)
# model.compile(loss='categorical_crossentropy', optimizer='adam')
# # pick a random seed
# start = numpy.random.randint(0, len(dataX)-1)
# pattern = dataX[start]
# print ("Seed:")
# print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# # generate characters
#