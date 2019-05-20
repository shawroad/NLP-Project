"""

@file   : 009-doc2vec句向量的训练.py

@author : xiaolu

@time1  : 2019-05-20

"""

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import jieba


def load_data():
    # 加载语料  并进行分词
    sentence = []
    with open('./data/text.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()    # 如果是训练句向量 这里一行代表一句话  如果这里训练的是文章向量 这里一行就代表一篇文章
        for line in lines:
            sentence.append(jieba.lcut(line))
    return sentence


def cal_vec(cut_word):

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(cut_word)]
    model = Doc2Vec(documents, vector_size=10, window=3, min_count=1, workers=4)
    # 1.dm=1 PV-DM  dm=0 PV-DBOW。
    # 2.size 所得向量的维度。
    # 3.window 上下文词语离当前词语的最大距离。
    # 4.alpha 初始学习率，在训练中会下降到min_alpha。
    # 5.min_count 词频小于min_count的词会被忽略。
    # 6.max_vocab_size 最大词汇表size，每一百万词会需要1GB的内存，默认没有限制。
    # 7.sample 下采样比例。
    # 8.iter 在整个语料上的迭代次数(epochs)，推荐10到20。
    # 9.hs=1 hierarchical softmax ，hs=0(default) negative sampling。
    # 10.dm_mean=0(default) 上下文向量取综合，dm_mean=1 上下文向量取均值。
    # 11.dbow_words:1训练词向量，0只训练doc向量。

    # 给定一个新文本   其实我乱写的这句跟语料没有一点关系，让计算机去找和语料那句话相似，针对难为计算机了
    text = '进行句向量的计算  按正常思路走，这里还需要对语料进行处理，如:去除停用次,去除低频次等'
    text = jieba.lcut(text)
    # 推断我给的那句话的向量
    vector = model.infer_vector(text)
    print(vector)   # 那句话的向量
    sims = model.docvecs.most_similar([vector], topn=3)
    print("输出与我们这句话最相似三句话:", sims)

    # 输出语料库中每句话对应的向量
    for i in range(len(cut_word)):
        print("第{}句话的向量为:".format(i))
        print(model.docvecs[i])


if __name__ == "__main__":
    cut_word = load_data()
    # print(cut_word)
    # print(len(cut_word))   # 10总共有十句话

    # 进行句向量的计算  按正常思路走，这里还需要对语料进行处理，如:去除停用次,去除低频次等
    cal_vec(cut_word)
