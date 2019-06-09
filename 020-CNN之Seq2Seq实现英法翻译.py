"""

@file   : 020-CNN之Seq2Seq实现英法翻译.py

@author : xiaolu

@time1  : 2019-06-09

"""
import numpy as np
from keras.layers import Input, Convolution1D, Dense
from keras.layers import Dot, Activation, Concatenate
from keras.models import Model

num_samples = 10000   # 用前10000个样本进行训练

input_texts = []    # 输入文本
target_texts = []   # 输出文本
input_characters = set()   # 输入字符表
target_charactrs = set()   # 输出字符表
with open('./data/fra.txt', 'r', encoding='utf8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # 使用\t作为文本的开始 \n作为文本的结束
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_charactrs:
            target_charactrs.add(char)

input_characters = sorted(list(input_characters))
target_charactrs = sorted(list(target_charactrs))

# 得到输入字符个数  和输出字符个数
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_charactrs)

# 获取每句话的长度
max_encoder_seq_length = max({len(txt) for txt in input_texts})
max_decoder_seq_length = max({len(txt) for txt in target_texts})

print('测试样本的个数:', len(input_texts))
print('输入样本的字符表大小:', num_encoder_tokens)
print('输出样本的字符表大小:', num_decoder_tokens)
print('输入样本中序列长度最长为:', max_encoder_seq_length)
print('输出样本中序列长度最长为:', max_decoder_seq_length)

# 字表和数字的映射
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)]
)
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_charactrs)]
)


encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')


# 把文本变为数字序列
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.

    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            decoder_target_data[i, t-1, target_token_index[char]] = 1.   # 解码的输出比输入总慢一拍 实现seq直接的预测



# 1.编码  使用了空洞卷积
encoder_inputs = Input(shape=(None, num_encoder_tokens))
x_encoder = Convolution1D(256, kernel_size=3, activation='relu', padding='causal')(encoder_inputs)
x_encoder = Convolution1D(256, kernel_size=3, activation='relu', padding='causal', dilation_rate=2)(x_encoder)
x_encoder = Convolution1D(256, kernel_size=3, activation='relu', padding='causal', dilation_rate=4)(x_encoder)

# 2.解码
decoder_inputs = Input(shape=(None, num_decoder_tokens))
x_decoder = Convolution1D(256, kernel_size=3, activation='relu', padding='causal')(decoder_inputs)
x_decoder = Convolution1D(256, kernel_size=3, activation='relu', padding='causal', dilation_rate=2)(x_decoder)
x_decoder = Convolution1D(256, kernel_size=3, activation='relu', padding='causal', dilation_rate=4)(x_decoder)

# 加入注意力机制
attention = Dot(axes=[2, 2])([x_decoder, x_encoder])  # 在第二个维度上进行对应相乘
attention = Activation('softmax')(attention)
context = Dot(axes=[2, 1])([attention, x_encoder])
decoder_combined_context = Concatenate(axis=-1)([context, x_decoder])

decoder_outputs = Convolution1D(64, kernel_size=3, activation='relu', padding='causal')(decoder_combined_context)
decoder_outputs = Convolution1D(64, kernel_size=3, activation='relu', padding='causal')(decoder_outputs)

# 输出
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

# 训练模型
batch = 64
epochs = 1

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data],
          decoder_target_data,
          batch_size=batch,
          epochs=epochs,
          validation_split=0.2)

model.save('cnn_seq2seq.h5')

# 为了使用模型，我们也加一个id到汉字的映射
recerse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items()
)
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items()
)

# 找100个样本进行验证
nb_examples = 100
in_encoder = encoder_input_data[:nb_examples]
in_decoder = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

# 给每句加开始的标志  其余维度都是0
in_encoder[:, 0, target_token_index['\t']] = 1

predict = np.zeros((len(input_texts), max_decoder_seq_length), dtype='float32')


for i in range(max_decoder_seq_length - 1):
    predict = model.predict([in_encoder, in_decoder])
    predict = predict.argmax(axis=-1)
    predict_ = predict[:, i].ravel().tolist()
    for j, x in enumerate(predict_):
        in_decoder[j, i+1, x] = 1


for seq_index in range(nb_examples):
    output_seq = predict[seq_index, :].ravel().tolist()
    decoded = []
    for x in output_seq:
        if reverse_target_char_index[x] == "\n":
            break
        else:
            decoded.append(reverse_target_char_index[x])
    decoded_sentence = "".join(decoded)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)



