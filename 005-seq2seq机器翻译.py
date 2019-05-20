"""

@file   : 005-seq2seq机器翻译.py

@author : xiaolu

@time1  : 2019-05-19

"""
# 语料的下载地址:http://www.manythings.org/anki/

from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
from keras.utils import plot_model
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

n_units = 256
batch = 64
epoch = 200
num_samples = 10000


# 1.加载文本
data_path = './data/cmn.txt'
df = pd.read_table(data_path, header=None).iloc[:num_samples, :]
df.columns = ['inputs', 'targets']
# print(df.head())
df['targets'] = df['targets'].apply(lambda x: '\t'+x+'\n')
# 将原文本和目标文本转为列表
input_texts = df.inputs.values.tolist()
target_texts = df.targets.values.tolist()
# print(input_texts[:10])
# print(target_texts[:10])


# 确定中英文各自包含的字符。df.unique()直接取sum可将unique数组中的各个句子拼接成一个长句子
input_characters = sorted(list(set(df.inputs.unique().sum())))     # 输入文本中都有哪些字符
target_characters = sorted(list(set(df.targets.unique().sum())))   # 输出文本中都有哪些字符
# print(input_characters)
print(target_characters)

input_length = max([len(i) for i in input_texts])
output_length = max([len(i) for i in target_texts])
input_feature_length = len(input_characters)
output_feature_length = len(target_characters)


# 2. 向量化   并将特征映射成数字
encoder_input = np.zeros((num_samples, input_length, input_feature_length))
decoder_input = np.zeros((num_samples, output_length, output_feature_length))
decoder_output = np.zeros((num_samples, output_length, output_feature_length))

input_dict = {c: index for index, c in enumerate(input_characters)}
input_dict_reverse = {index: c for index, c in enumerate(input_characters)}
target_dict = {c: index for index, c in enumerate(target_characters)}
target_dict_reverse = {index: c for index, c in enumerate(target_characters)}

# 将每天句子转为one_hot编码  [[[], [], [], [], [], []..这是一个句子的one_hot], [[], [], [], []..这是一个句子的one-hot]...]
for seq_index, seq in enumerate(input_texts):
    for char_index, char in enumerate(seq):
        encoder_input[seq_index, char_index, input_dict[char]] = 1.0

for seq_index,  seq in enumerate(target_texts):
    for char_index, char in enumerate(seq):
        decoder_input[seq_index, char_index, target_dict[char]] = 1.0
        if char_index > 0:
            decoder_output[seq_index, char_index-1, target_dict[char]] = 1.0

# 看一下向量化的数据
eng = ''.join([input_dict_reverse[np.argmax(i)] for i in encoder_input[0] if max(i) != 0])
hanzi = ''.join([target_dict_reverse[np.argmax(i)] for i in decoder_output[0] if max(i) != 0])
hanzi2 = ''.join([target_dict_reverse[np.argmax(i)] for i in decoder_input[0] if max(i) != 0])
# print(eng)
# print(hanzi)
# print(hanzi2)


# 建立模型
def create_model(n_input, n_output, n_units):
    # 一:训练阶段
    # 1.编码
    encoder_input = Input(shape=(None, n_input))
    # n_units为LSTM单元中每个门的神经元的个数，return_state设为True时才会返回最后时刻的状态h,c
    encoder = LSTM(n_units, return_state=True)
    _, encoder_h, encoder_c = encoder(encoder_input)
    encoder_state = [encoder_h, encoder_c]   # 保留encoder的末状态作为decoder的初始状态

    # 2.解码
    decoder_input = Input(shape=(None, n_output))
    # 训练的解码需要decoder的输出序列来与结果对比优化，故return_sequences也要设为True
    decoder = LSTM(n_units, return_sequences=True, return_state=True)
    # 在训练阶段只需要用到decoder的输出序列，不需要用最终状态h.c
    decoder_output, _, _ = decoder(decoder_input, initial_state=encoder_state)   # 这里的初始化是编码的输出
    # 输出序列经过全连接层得到结果
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_output = decoder_dense(decoder_output)
    # 3. 将模型串起来
    # 第一个参数为训练模型的输入，包含了encoder和decoder的输入，第二个参数为模型的输出，包含了decoder的输出
    model = Model([encoder_input, decoder_input], decoder_output)

    # 二: 推理阶段
    # 1.推理的编码阶段
    encoder_infer = Model(encoder_input, encoder_state)
    # 2.推理的解码阶段
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    # 上个时刻的状态h,c
    decoder_state_input = [decoder_state_input_h, decoder_state_input_c]
    decoder_infer_output, decoder_infer_state_h, decoder_infer_state_c = decoder(decoder_input, initial_state=decoder_state_input)
    decoder_infer_state = [decoder_infer_state_h, decoder_infer_state_c]  # 当前时刻得到的状态
    decoder_infer_output = decoder_dense(decoder_infer_output)   # 当前时刻的输出
    decoder_infer = Model([decoder_input]+decoder_state_input, [decoder_infer_output]+decoder_infer_state)

    return model, encoder_infer, decoder_infer


model_train, encoder_infer, decoder_infer = create_model(input_feature_length, output_feature_length, n_units)

# 查看模型的结构
# plot_model(to_file='model.png', model=model_train, show_shapes=True)
# plot_model(to_file='encoder.png', model=encoder_infer, show_shapes=True)
# plot_model(to_file='decoder.png', model=decoder_infer, show_shapes=True)

model_train.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# print(model_train.summary())
# print(encoder_infer.summary())
# print(decoder_infer.summary())

# 模型训练
model_train.fit([encoder_input, decoder_input], decoder_output, batch_size=batch, epochs=epoch, validation_split=0.2)


# 预测序列   predict_chinese(test, encoder_infer, decoder_infer, output_length, output_feature_length)
def predict_chinese(source, encoder_inference, decoder_inference, n_steps, features):
    # 推理的编码将原文本转为其隐态
    state = encoder_inference.predict(source)

    # 第一个字符'\t',为起始标志
    predict_seq = np.zeros((1, 1, features))   # 构造最后要输出的那个one_hot矩阵
    predict_seq[0, 0, target_dict['\t']] = 1

    output = ''
    # 开始对encoder获得的隐状态进行推理
    # 每次循环用上次预测的字符作为输入来预测下一次的字符，直到预测出了终止符
    for i in range(n_steps):  # n_steps为句子最大长度
        # 给decoder输入上一个时刻的h,c隐状态，以及上一次的预测字符predict_seq
        yhat, h, c = decoder_inference.predict([predict_seq]+state)
        # 注意，这里的yhat为Dense之后输出的结果，因此与h不同
        char_index = np.argmax(yhat[0, -1, :])
        char = target_dict_reverse[char_index]
        output += char
        state = [h, c]  # 本次状态做为下一次的初始状态继续传递
        predict_seq = np.zeros((1, 1, features))
        predict_seq[0, 0, char_index] = 1
        if char == '\n':  # 预测到了终止符则停下来
            break
    return output


for i in range(1000, 1100):
    test = encoder_input[i:i+1, :, :]  # i:i+1保持数组是三维
    out = predict_chinese(test, encoder_infer, decoder_infer, output_length, output_feature_length)
    # print(input_texts[i],'\n---\n',target_texts[i],'\n---\n',out)
    print(input_texts[i])
    print(out)
