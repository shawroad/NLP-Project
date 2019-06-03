"""

@file   : 017-LSTM+Greedy实现英文单词纠错.py

@author : xiaolu

@time1  : 2019-06-03

"""
from tensorflow.python.layers.core import Dense
import tensorflow as tf
import numpy as np


# 网络结构参数配置
class ModelConfig:
    encoder_hidden_layers = [50, 50]   # 编码过程中 深度为两层 每层的输出为50维
    decoder_hidden_layers = [50, 50]   # 解码过程中 深度为两层 每层的输出为50维
    dropout_prob = 0.5
    encoder_embedding_size = 15  # 词嵌入维度
    decoder_embedding_size = 15  # 词嵌入维度


# 训练参数的配置
class TrainConfig:
    epochs = 10
    every_checkpoint = 1000
    learning_rate = 0.01
    max_grad_norm = 3


class Config:
    batch_size = 128
    infer_prob = 0.2

    source_path = "data/letters_source.txt"
    target_path = "data/letters_target.txt"

    train = TrainConfig()
    model = ModelConfig()


config = Config()


# 生成数据
class DataGen:
    def __init__(self, config):
        self.source_path = config.source_path
        self.target_path = config.target_path

        # 建立字符到id的映射  分原文本和目标文本
        self.source_char_to_int = {}
        self.source_int_to_char = {}
        self.target_char_to_int = {}
        self.target_int_to_char = {}

        self.source_data = []
        self.target_data = []

    def read_data(self):
        # 读源数据
        with open(self.source_path, 'r', encoding='utf8') as f:
            source_char_to_int, source_int_to_char, source_data = self.gen_vocab_dict(f.read())

        self.source_char_to_int = source_char_to_int
        self.source_int_to_char = source_int_to_char
        self.source_data = source_data

        # 读目标数据
        with open(self.target_path, 'r', encoding='utf8') as f:
            target_char_to_int, target_int_to_char, target_data = self.gen_vocab_dict(f.read(), True)

        self.target_char_to_int = target_char_to_int
        self.target_int_to_char = target_int_to_char
        self.target_data = target_data

    def gen_vocab_dict(self, string, is_target=False):
        # 生成词典

        special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']

        # 1.建立字符到id的映射
        vocab = list(set(string))    # 将读进来的文本当做一个大的字符表 然后去重
        vocab.remove('\n')
        vocab = special_words + vocab
        int_to_char = {index: char for index, char in enumerate(vocab)}
        char_to_int = {char: index for index, char in int_to_char.items()}

        # 2.遍历每个词 然后将词转为id序列  目标词记得加EOS
        word_list = string.strip().split("\n")   # 词列表
        if is_target:
            data = [[char_to_int.get(char, '<UNK>') for char in word] + [char_to_int['<EOS>']] for word in word_list]
        else:
            data = [[char_to_int.get(char, '<UNK>') for char in word] for word in word_list]

        return char_to_int, int_to_char, data


dataGen = DataGen(config)
dataGen.read_data()

# 看一下源数据和目标数据的第一条
print("源数据:", dataGen.source_data[0])    # 源数据: [21, 13, 20, 23, 23]
print("目标数据:", dataGen.target_data[0])   # 目标数据: [20, 21, 23, 23, 13, 3]

print("源数据条数:", len(dataGen.source_data))   # 源数据条数: 10000
print("目标数据条数:", len(dataGen.target_data))  # 目标数据条数: 10000


# 定义模型
class Seq2SeqModel():
    def __init__(self, config, encoder_vocab_size, target_char_to_int, is_infer=False):
        # config: 最上面两个类中的参数
        # encoder_vocab_size: 词嵌入的维度
        # target_char_to_int: 字符到id的映射
        self.inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
        self.targets = tf.placeholder(tf.int32, [None, None], name="targets")

        self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")

        self.source_sequence_length = tf.placeholder(tf.int32, [None], name="source_sequence_length")
        self.target_sequence_length = tf.placeholder(tf.int32, [None], name="target_sequence_length")

        # tf.reduce_max函数的作用：计算张量的各个维度上的元素的最大值
        self.target_max_length = tf.reduce_max(self.target_sequence_length, name='target_max_length')

        decoder_output = self.seq2seq(config, encoder_vocab_size, target_char_to_int, is_infer)

        if is_infer:
            self.infer_logits = tf.identity(decoder_output.sample_id, "infer_logits")

        else:
            self.logits = tf.identity(decoder_output.rnn_output, "logits")

            masks = tf.sequence_mask(self.target_sequence_length, self.target_max_length, dtype=tf.float32, name="mask")

            with tf.name_scope("loss"):   # 计算损失  带有mask
                self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.targets, masks)

            with tf.name_scope("accuracy"):
                # 计算准确率
                self.predictions = tf.argmax(self.logits, 2)
                correctness = tf.equal(tf.cast(self.predictions, dtype=tf.int32), self.targets)
                self.accu = tf.reduce_mean(tf.cast(correctness, "float"), name="accu")

    # 编码的过程
    def encoder(self, config, encoder_vocab_size):
        encoder_embed_input = tf.contrib.layers.embed_sequence(self.inputs, encoder_vocab_size,
                                                               config.model.encoder_embedding_size)

        def get_lstm_cell(hidden_size):
            # 添加LSTM 并初始化
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True,
                                                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))

            # 使用 DropoutWrapper 类来实现 dropout 功能，input_keep_prob 控制输出的 dropout 概率
            drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.dropout_prob)
            return drop_cell

        # 构造两层的LSMT
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [get_lstm_cell(hidden_size) for hidden_size in config.model.encoder_hidden_layers])   # [50, 50]
        # 相当于将构造的结构搭建起来
        outputs, final_state = tf.nn.dynamic_rnn(cell, encoder_embed_input, sequence_length=self.source_sequence_length,
                                                 dtype=tf.float32)
        return outputs, final_state


    # 解码的过程
    def decoder(self, config, encoder_state, target_char_to_int, is_infer):

        decoder_vocab_size = len(target_char_to_int)

        # 对解码的输入进行字嵌入
        embeddings = tf.Variable(tf.random_uniform([decoder_vocab_size, config.model.decoder_embedding_size]))
        decoder_embed_input = tf.nn.embedding_lookup(embeddings, self.targets)

        # 和编码的过程一样 建立两层的LSTM
        def get_lstm_cell(hidden_size):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True,
                                                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.dropout_prob)

            return drop_cell

        # 将定义解码的结构搭建起来
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [get_lstm_cell(hidden_size) for hidden_size in config.model.decoder_hidden_layers])

        # 定义有Dense方法生成的全连接层
        output_layer = Dense(decoder_vocab_size,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        # 定义训练时的decode的代码
        with tf.variable_scope("decode"):
            # 得到help对象，帮助读取数据
            train_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                             sequence_length=self.target_sequence_length)

            # 构建decoder  传入 网络结构, 解码的输入, 编码的状态, 输出
            train_decoder = tf.contrib.seq2seq.BasicDecoder(cell, train_helper, encoder_state, output_layer)
            train_decoder_output, train_state, train_sequence_length = tf.contrib.seq2seq.dynamic_decode(train_decoder,
                                                                                                         impute_finished=True,
                                                                                                         maximum_iterations=self.target_max_length)

        # 定义预测时的decode代码  reuse置为True
        with tf.variable_scope("decode", reuse=True):
            # 解码时的第一个时间步上的输入，之后的时间步上的输入是上一时间步的输出
            start_tokens = tf.tile(tf.constant([target_char_to_int["<GO>"]], dtype=tf.int32), [config.batch_size],
                                   name="start_tokens")

            # 解码时按贪心法解码，按照最大条件概率来预测输出值，该方法需要输入启动词和结束词，启动词是个一维tensor，结束词是标量
            infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens,
                                                                    target_char_to_int["<EOS>"])
            infer_decoder = tf.contrib.seq2seq.BasicDecoder(cell, infer_helper, encoder_state, output_layer)
            infer_decoder_output, infer_state, infer_sequence_length = tf.contrib.seq2seq.dynamic_decode(infer_decoder,
                                                                                                         impute_finished=True,
                                                                                                         maximum_iterations=self.target_max_length)

        if is_infer:
            return infer_decoder_output

        return train_decoder_output

    def seq2seq(self, config, encoder_vocab_size, target_char_to_int, is_infer):
        """
        将encoder和decoder合并输出
        """
        encoder_output, encoder_state = self.encoder(config, encoder_vocab_size)

        decoder_output = self.decoder(config, encoder_state, target_char_to_int, is_infer)

        return decoder_output


# 将数据pad成同样的长度
def pad_batch(batch, char_to_int):
    sequence_length = [len(sequence) for sequence in batch]
    max_length = max(sequence_length)
    # 将数据pad成同样的长度
    new_batch = [sequence + [char_to_int["<PAD>"]] * (max_length - len(sequence)) for sequence in batch]

    return sequence_length, max_length, new_batch


def next_batch(source, target, batch_size, source_char_to_int, target_char_to_int):
    # 制造批数据
    num_batches = len(source) // batch_size
    for i in range(num_batches):
        source_batch = source[i * batch_size: (i + 1) * batch_size]
        target_batch = target[i * batch_size: (i + 1) * batch_size]

        source_sequence_length, source_max_length, new_source_batch = pad_batch(source_batch, source_char_to_int)
        target_sequence_length, target_max_length, new_target_batch = pad_batch(target_batch, target_char_to_int)

        yield dict(source_batch=np.array(new_source_batch),
                   target_batch=np.array(new_target_batch),
                   source_sequence_length=np.array(source_sequence_length),
                   target_sequence_length=np.array(target_sequence_length),
                   target_max_length=target_max_length)


# 训练模型
class Engine:
    def __init__(self):
        self.config = Config()
        self.dataGen = DataGen(self.config)
        self.dataGen.read_data()
        self.sess = None
        self.global_step = 0

    def train_step(self, sess, train_op, train_model, params):

        feed_dict = {
            train_model.inputs: params["source_batch"],
            train_model.targets: params["target_batch"],
            train_model.dropout_prob: self.config.model.dropout_prob,
            train_model.source_sequence_length: params["source_sequence_length"],
            train_model.target_sequence_length: params["target_sequence_length"],
        }

        _, loss, accu = sess.run([train_op, train_model.loss, train_model.accu], feed_dict)

        return loss, accu

    def infer_step(self, sess, infer_model, params):

        feed_dict = {
            infer_model.inputs: params["source_batch"],
            infer_model.targets: params["target_batch"],
            infer_model.dropout_prob: 1.0,
            infer_model.source_sequence_length: params["source_sequence_length"],
            infer_model.target_sequence_length: params["target_sequence_length"],
        }

        logits = sess.run([infer_model.infer_logits], feed_dict)
        predictions = logits[0]
        # 预测的结果
        prediction = [sequence[:end] for sequence in predictions for end in params["target_sequence_length"]]
        # 真实结果
        target = [sequence[:end] for sequence in params["target_batch"] for end in params["target_sequence_length"]]

        # 计算准确率
        total = 0
        correct = 0
        for i in range(len(prediction)):
            for j in range(len(prediction[i])):
                if prediction[i][j] == target[i][j]:
                    correct += 1
            total += len(prediction[i])

        accu = correct / total

        return accu

    def run_epoch(self):
        config = self.config
        dataGen = self.dataGen

        source_data = dataGen.source_data
        target_data = dataGen.target_data

        # 训练数据集和测试数据集的切分比例
        train_split = int(len(source_data) * config.infer_prob)

        # 切分源数据
        train_source_data = source_data[train_split:]
        infer_source_data = source_data[: train_split]

        # 切分目标数据
        train_target_data = target_data[train_split:]
        infer_target_data = target_data[: train_split]

        source_char_to_int = dataGen.source_char_to_int
        target_char_to_int = dataGen.target_char_to_int

        encoder_vocab_size = len(source_char_to_int)

        batch_size = config.batch_size

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                with tf.name_scope("train"):
                    with tf.variable_scope("seq2seq"):
                        train_model = Seq2SeqModel(config, encoder_vocab_size, target_char_to_int, is_infer=False)

                with tf.name_scope("infer"):
                    with tf.variable_scope("seq2seq", reuse=True):
                        infer_model = Seq2SeqModel(config, encoder_vocab_size, target_char_to_int, is_infer=True)

                global_step = tf.Variable(0, name="global_step", trainable=False)

                optimizer = tf.train.AdamOptimizer(config.train.learning_rate)
                grads_and_vars = optimizer.compute_gradients(train_model.loss)
                grads_and_vars = [(tf.clip_by_norm(g, config.train.max_grad_norm), v) for g, v in grads_and_vars if
                                  g is not None]
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name="train_op")

                saver = tf.train.Saver(tf.global_variables())
                sess.run(tf.global_variables_initializer())

                print("初始化完成，开始训练模型")
                for i in range(config.train.epochs):
                    for params in next_batch(train_source_data, train_target_data, batch_size, source_char_to_int,
                                             target_char_to_int):
                        loss, accu = self.train_step(sess, train_op, train_model, params)
                        current_step = tf.train.global_step(sess, global_step)
                        print("step: {}  loss: {}  accu: {}".format(current_step, loss, accu))

                        if current_step % config.train.every_checkpoint == 0:
                            accus = []
                            for params in next_batch(infer_source_data, infer_target_data, batch_size,
                                                     source_char_to_int, target_char_to_int):
                                accu = self.infer_step(sess, infer_model, params)
                                accus.append(accu)
                            print("\n")
                            print("Evaluation accuracy: {}".format(sum(accus) / len(accus)))
                            print("\n")
                            saver.save(sess, "model/my-model", global_step=current_step)


engine = Engine()
engine.run_epoch()










