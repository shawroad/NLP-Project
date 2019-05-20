"""

@file   : 004-实现一个关键词提取算法.py

@author : xiaolu

@time1  : 2019-05-08

"""
import jieba
import jieba.posseg as psg
import math
from gensim import corpora, models
from jieba import analyse
import functools
import numpy as np


# 加载停用词
def get_stopword_list():
    # 读取停用词表
    stop_word_path = './data/stopword.txt'
    stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path, 'r', encoding='utf8').readlines()]
    return stopword_list


# 分词
def seg_to_list(sentence, pos=False):
    # 这里是进行分词  pas代表分词要不要标注词性
    if not pos:
        # 不进行词性标注的分词
        seg_list = jieba.cut(sentence)
    else:
        # 进行词性标注的分词
        seg_list = psg.cut(sentence)
    return seg_list


# 剔除干扰词（词性不符合或者是停用词）最后留下来的是比较符合要求的词
def word_filter(seg_list, pos=False):
    stopword_list = get_stopword_list()
    filter_list = []
    for seg in seg_list:
        if not pos:  # 如不过滤词性，我们将所有词的词性标记为名词
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):   # 这里只保留名词
            continue
        # 过滤停用词表中的词，以及长度为<2的词
        if not word in stopword_list and len(word) > 1:
            filter_list.append(word)

    return filter_list


# 加载语料库，pos为是否词性标注的参数，corpus_path为数据集路径
def load_data(pos=False, corpus_path='./data/corpus.txt'):

    doc_list = []
    for line in open(corpus_path, 'r', encoding='utf8'):
        content = line.strip()  # 这里一行代表一个文本
        seg_list = seg_to_list(content, pos)
        filter_list = word_filter(seg_list, pos)
        doc_list.append(filter_list)   # [[文章一中的词, 词1, 词2,....], [文章二中的词], [文章三中的词]...]

    return doc_list


# idf值统计方法
def train_idf(doc_list):
    idf_dic = {}
    # 总文档数
    tt_count = len(doc_list)

    # 每个词出现的文档数
    for doc in doc_list:
        for word in set(doc):
            if word in idf_dic.keys():
                idf_dic[word] += 1.0
            else:
                idf_dic[word] = 1.0
            # idf_dic[word] = idf_dic.get(word, 0.0) + 1.0

    # 按公式转换为idf值，分母加1进行平滑处理
    for k, v in idf_dic.items():
        idf_dic[k] = math.log(tt_count / (1.0 + v))  # 计算出每个词的idf值

    # 对于没有在字典中的词，默认其仅在一个文档出现，得到默认idf值
    default_idf = math.log(tt_count / (1.0))
    return idf_dic, default_idf


# 排序函数   # 主要考虑当前两个词要不要合并
def cmp(e1, e2):
    # e[0] 应该是词， e[1] 代表的是tfidf值
    res = np.sign(e1[1] - e2[1])  # np.sign(x) x式子如果是负值则结果为-1   x式子的值为0 则结果为0  x式子的值为正 则结果为1
    if res != 0:
        return res
    else:    # 如果前后两个词的tf-idf值相同，我们将其合并
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1


# TF-IDF类
class TfIdf(object):
    # 四个参数分别是：训练好的idf字典，默认idf值，处理后的待提取文本，关键词数量
    def __init__(self, idf_dic, default_idf, word_list, keyword_num):
        self.word_list = word_list   # 待提取的文本
        self.idf_dic, self.default_idf = idf_dic, default_idf   # 训练好的idf  默认的idf
        self.tf_dic = self.get_tf_dic()
        self.keyword_num = keyword_num   # 关键词数量

    # 统计tf值
    def get_tf_dic(self):
        tf_dic = {}
        for word in self.word_list:
            if word in tf_dic.keys():
                tf_dic[word] += 1.0
            else:
                tf_dic[word] = 1.0
            # tf_dic[word] = tf_dic.get(word, 0.0) + 1.0   # 统计每个词出现的次数

        tt_count = len(self.word_list)  # 统计文本中词汇的总数量
        for k, v in tf_dic.items():
            tf_dic[k] = float(v) / tt_count   # 这个词在文章中出现的概率

        return tf_dic

    # 按公式计算tf-idf
    def get_tfidf(self):

        tfidf_dic = {}
        for word in self.word_list:
            idf = self.idf_dic.get(word, self.default_idf)
            tf = self.tf_dic.get(word, 0)

            tfidf = tf * idf
            tfidf_dic[word] = tfidf

        tfidf_dic.items()   # tfidf_dic 格式是{"词语":对应的tfidf值}
        # 根据tf-idf排序，去排名前keyword_num的词作为关键词
        for k, v in sorted(tfidf_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            print(k + "/ ", end='')  # 我们只需输出对应的词  不需要tfidf值
        print()


def tfidf_extract(word_list, pos=False, keyword_num=10):

    doc_list = load_data(pos)
    idf_dic, default_idf = train_idf(doc_list)  # 注意:这里加载获取的是语料库中的idf值
    tfidf_model = TfIdf(idf_dic, default_idf, word_list, keyword_num)
    tfidf_model.get_tfidf()


# textrank算法
def textrank_extract(text, pos=False, keyword_num=10):
    textrank = analyse.textrank
    keywords = textrank(text, keyword_num)   # text是一个列表 [词1, 词2, 词2....] 这些词是过滤后的词
    # 输出抽取出的关键词
    for keyword in keywords:
        print(keyword + "/ ", end='')
    print()


# 主题模型
class TopicModel(object):
    # 三个传入参数：处理后的数据集，关键词数量，具体模型（LSI、LDA），主题数量  这里的主题数量相当于kmeans中要指定聚为几类
    def __init__(self, doc_list, keyword_num, model='LSI', num_topics=4):
        # 使用gensim的接口，将文本转为向量化表示
        # 先构建词空间
        self.dictionary = corpora.Dictionary(doc_list)

        # 使用BOW模型向量化
        # corpus的格式为: [[(词编号, 在当前句子出现第几次)，(*), (*), (*)], [*], [*].....]
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]

        # 对每个词，根据tf-idf进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]   # corpus_tfidf 的格式 [[(词标号, 对应的tfidf值), (*), (*)], []]

        self.keyword_num = keyword_num
        self.num_topics = num_topics
        # 选择加载的模型
        if model == 'LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()

        # 得到数据集的主题-词分布
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    def train_lsi(self):
        lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lsi

    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lda

    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}

        for word in word_dic:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    # 计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词
    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

        # 余弦相似度计算
        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim

        # 计算输入文本和每个词的主题分布相似度
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = calsim(v, senttopic)
            sim_dic[k] = sim

        for k, v in sorted(sim_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            print(k + "/ ", end='')
        print()

    # 词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法
    def word_dictionary(self, doc_list):
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)

        dictionary = list(set(dictionary))

        return dictionary

    def doc2bowvec(self, word_list):
        vec_list = [1 if word in word_list else 0 for word in self.dictionary]
        return vec_list


def topic_extract(word_list, model, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    topic_model = TopicModel(doc_list, keyword_num, model=model)
    topic_model.get_simword(word_list)


if __name__ == '__main__':
    text = '6月19日,《2012年度“中国爱心城市”公益活动新闻发布会》在京举行。' + \
           '中华社会救助基金会理事长许嘉璐到会讲话。基金会高级顾问朱发忠,全国老龄' + \
           '办副主任朱勇,民政部社会救助司助理巡视员周萍,中华社会救助基金会副理事长耿志远,' + \
           '重庆市民政局巡视员谭明政。晋江市人大常委会主任陈健倩,以及10余个省、市、自治区民政局' + \
           '领导及四十多家媒体参加了发布会。中华社会救助基金会秘书长时正新介绍本年度“中国爱心城' + \
           '市”公益活动将以“爱心城市宣传、孤老关爱救助项目及第二届中国爱心城市大会”为主要内容,重庆市' + \
           '、呼和浩特市、长沙市、太原市、蚌埠市、南昌市、汕头市、沧州市、晋江市及遵化市将会积极参加' + \
           '这一公益活动。中国雅虎副总编张银生和凤凰网城市频道总监赵耀分别以各自媒体优势介绍了活动' + \
           '的宣传方案。会上,中华社会救助基金会与“第二届中国爱心城市大会”承办方晋江市签约,许嘉璐理' + \
           '事长接受晋江市参与“百万孤老关爱行动”向国家重点扶贫地区捐赠的价值400万元的款物。晋江市人大' + \
           '常委会主任陈健倩介绍了大会的筹备情况。'
    pos = True
    seg_list = seg_to_list(text, pos)
    filter_list = word_filter(seg_list, pos)

    print('TF-IDF模型结果:')
    tfidf_extract(filter_list)

    print("textrank模型结果:")
    textrank_extract(text, pos=False, keyword_num=10)

    print('LSI模型结果：')
    topic_extract(filter_list, 'LSI', pos)

    print('LDA模型结果：')
    topic_extract(filter_list, 'LDA', pos)







