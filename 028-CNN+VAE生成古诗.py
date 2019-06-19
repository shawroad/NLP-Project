"""

@file   : 028-CNN+VAE生成古诗.py

@author : xiaolu

@time   : 2019-06-19

"""
from keras.layers import Layer
import numpy as np
import keras.backend as K
from keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
from keras.models import Model
from keras.layers import Lambda, Reshape
from keras.callbacks import Callback

def load_data(path):
    # 只读5言绝句
    with open(path, 'r', encoding='utf8') as f:
        five_poetry = []
        lines = f.readlines()
        # print(lines[0])
        # print(len(lines))    # 43030
        for line in lines:
            temp = line.replace('\n', '').split('。')
            for t in temp:
                if len(t) == 11:
                    five_poetry.append(t)
    return five_poetry


def process_data(data):
    # 建立字表
    voacb = set()
    for d in data:
        for i in list(d):
            voacb.update(i)
    # print(len(voacb))    # 6875

    vocab = list(voacb)
    vocab2id = {w: i for i, w in enumerate(vocab)}
    id2vocab = {i: w for w, i in vocab2id.items()}

    # 将每首诗转为id序列
    x = [[vocab2id.get(w) for w in list(d)] for d in data]
    return x, vocab2id, id2vocab


class GCNN(Layer):
    def __init__(self, output_dim=None, residual=False, **kwargs):
        super(GCNN, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.residual = residual

    def build(self, input_shape):
        if self.output_dim == None:
            self.output_dim = input_shape[-1]

        self.kernel = self.add_weight(name='gcc_kernel',
                                      shape=(3, input_shape[-1], self.output_dim*2),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, x, **kwargs):
        _ = K.conv1d(x, self.kernel, padding='same')
        _ = _[:, :, :self.output_dim] * K.sigmoid(_[:, :, self.output_dim:])   # 相当于带点门限机制  看那些需要哪些不需要
        if self.residual:
            return _ + x
        else:
            return _


def sampling(args):
    # 采样
    latent_dim = 64
    z_mean, z_log_var = args
    # 先进行标准正太分布采样
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
    # 然后将采样的样本转为非标准正太分布的样本
    return z_mean + K.exp(z_log_var / 2) * epsilon


def build_model():
    # 定义模型
    n = 5  # 只抽取五言诗
    latent_dim = 64  # 隐变量维度
    hidden_dim = 64  # 隐层节点数

    # 定义编码
    input_sentence = Input(shape=(2*n+1,), dtype='int32')        # (None, 11)
    input_vec = Embedding(len(vocab2id), hidden_dim)(input_sentence)     # (None, 11, 64)
    h = GCNN(residual=True)(input_vec)   # (None, 11, 64)
    h = GCNN(residual=True)(h)       # (None, 11, 64)
    h = GlobalAveragePooling1D()(h)    # (None, 64)

    # 算均值和方差
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    # 给均值和方差  让其去采样
    z = Lambda(sampling)([z_mean, z_log_var])

    # 定义解码
    decoder_hidden = Dense(hidden_dim * (2 * n + 1))
    decoder_cnn = GCNN(residual=True)
    decoder_dense = Dense(len(vocab2id), activation='softmax')

    h = decoder_hidden(z)
    h = Reshape((2*n+1, hidden_dim))(h)
    h = decoder_cnn(h)
    output = decoder_dense(h)

    # 建立模型
    vae = Model(input_sentence, output)

    # 定义损失   重构损失+KL损失
    xent_loss = K.sum(K.sparse_categorical_crossentropy(input_sentence, output), 1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    # add_loss是新增的方法，用于更灵活地添加各种loss
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()

    # 重用解码层，构建单独的生成模型
    decoder_input = Input(shape=(latent_dim,))
    _ = decoder_hidden(decoder_input)
    _ = Reshape((2 * n + 1, hidden_dim))(_)
    _ = decoder_cnn(_)
    _output = decoder_dense(_)
    generator = Model(decoder_input, _output)

    # 利用生成模型随机生成一首诗
    def gen():
        latent_dim = 64
        n = 5
        r = generator.predict(np.random.randn(1, latent_dim))[0]
        r = r.argmax(axis=1)
        return ''.join([id2vocab[i] for i in r[: 2*n+1]])

    # 回调器，方便在训练过程中输出
    class Evaluate(Callback):
        def __init__(self):
            super(Evaluate, self).__init__()
            self.log = []

        def on_epoch_end(self, epoch, logs=None):
            self.log.append(gen())
            print(u'          %s'%(self.log[-1])).encode('utf-8')

    evaluator = Evaluate()

    vae.fit(x,
            shuffle=True,
            epochs=100,
            batch_size=64,
            callbacks=[evaluator])

    vae.save_weights('shi.model')

    for i in range(20):
        print(gen())


if __name__ == "__main__":
    path = './data/poetry.txt'
    data = load_data(path)
    print(data[:5])
    print(len(data))    # 147077

    # 整理字表，并将古诗映射成id
    x, vocab2id, id2vocab = process_data(data)
    x = np.array(x)

    # 定义模型
    build_model()

