"""

@file   : 04-word2vec.py

@author : xiaolu

@time1  : 2019-05-19

"""
import jieba
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence


def load_data():
    # 加载预料   这语料自己爬一些中文文本就可以了。   
    with open('./data/corpus.txt', 'r', encoding='utf8') as f:
        text = f.read()
        text = text.strip().replace('\n', '')
    return text


def split_word(text): 
    # 进行分词   这里按正常流程走应该还得去停用词，标点符号等。 我省略了。真正项目中得去停用词
    word_list = jieba.lcut(text)
    text = ' '.join(word_list)
    text = text.encode('utf8')
    with open('./data/corpus_new.txt', 'wb') as f:
        f.write(text)   # 记住 这里面存的是分词后组成的文本



def train_vec():

    model = Word2Vec(LineSentence('./data/corpus_new.txt'), hs=1, min_count=1, window=3, size=50)
    # sentences: 我们要分析的语料，可以是一个列表，或者从文件中遍历读出。后面我们会有从文件读出的例子。
    # size: 词向量的维度，默认值是100。这个维度的取值一般与我们的语料的大小相关，如果是不大的语料，比如小于100M的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度。
    # window：即词向量上下文最大距离，这个参数在我们的算法原理篇中标记为，window越大，则和某一词较远的词也会产生上下文关系。默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5, 10]之间。
    # sg: 即我们的word2vec两个模型的选择了。如果是0， 则是CBOW模型，是1则是Skip - Gram模型，默认是0即CBOW模型。
    # hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative
    # Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical
    # Softmax。默认是0即Negative
    # Sampling。
    # negative: 即使用Negative
    # Sampling时负采样的个数，默认是5。推荐在[3, 10]之间。这个参数在我们的算法原理篇中标记为neg。
    # cbow_mean: 仅用于CBOW在做投影的时候，为0，则算法中的为上下文的词向量之和，为1则为上下文的词向量的平均值。默认值也是1，不推荐修改默认值。
    # min_count: 需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值。
    # iter: 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。
    # alpha: 在随机梯度下降法中迭代的初始步长。算法原理篇中标记为，默认是0.025。
    # min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每轮的迭代步长可以由iter，alpha， min_alpha一起得出。这部分由于不是word2vec算法的核心内容。对于大语料，需要对alpha, min_alpha, iter一起调参，来选择合适的三个值。

    # 计算体育与中国篮协的距离
    sim = model.similarity('体育', '中国篮协')
    print("体育与中国篮协的距离:", sim)
    #
    # # 计算与体育最近的三个词
    word = model.most_similar('体育', topn=3)
    print("与体育最接近的词:", word)
    #
    # 输出体育的词向量
    print("体育的词向量:", model['体育'])

    # 保存模型
    model.save("gensim实现word2vec.model")
    # 加载模型
    # model = Word2Vec.load('gensim实现word2vec.model')


if __name__ == '__main__':
    # 加载预料
    text = load_data()
    # 分词
    split_word(text)
    # 训练模型
    train_vec()
