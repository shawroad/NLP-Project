"""

@file   : 008-用kmeans对豆瓣读书数据聚类.py

@author : xiaolu

@time1  : 2019-05-11

"""
import re
import string
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter


# ------------------去掉无用字符，去除停用词等------------------------
# 加载停用词
with open("data/stop_words.utf8", encoding="utf8") as f:
    stopword_list = f.readlines()


def tokenize_text(text):
    tokens = jieba.lcut(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ''.join(filtered_tokens)
    return filtered_text


def normalize_corpus(corpus):
    normalized_corpus = []
    for text in corpus:

        text = " ".join(jieba.lcut(text))
        normalized_corpus.append(text)

    return normalized_corpus

# -----------------------------------------------------------------------


def build_feature_matrix(documents, feature_type='frequency', ngram_range=(1, 1), min_df=0.0, max_df=1.0):
    # 将文本转成向量
    feature_type = feature_type.lower().strip()

    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise Exception("滚，你输入的转换类型有问题。")
    feature_matrix = vectorizer.fit_transform(documents).astype(float)

    return vectorizer, feature_matrix


def k_means(feature_matrix, num_clusters=10):
    # 进行聚类。指定聚为10类
    km = KMeans(n_clusters=num_clusters, max_iter=10000)
    km.fit(feature_matrix)
    clusters = km.labels_
    return km, clusters


def get_cluster_data(clustering_obj, book_data, feature_names, num_clusters, topn_features=10):

    cluster_details = {}
    # 获取每个簇的中心点,然后得到它在各个特征的值
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]

    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        # 获取每个簇中的文本数量
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        # 获取前多少个重要的特征
        key_features = [feature_names[index] for index in ordered_centroids[cluster_num, :topn_features]]

        cluster_details[cluster_num]['key_features'] = key_features

        books = book_data[book_data['Cluster'] == cluster_num]['title'].values.tolist()

        cluster_details[cluster_num]['books'] = books

    return cluster_details


def print_cluster_data(cluster_data):
    # print cluster details
    for cluster_num, cluster_details in cluster_data.items():
        print('Cluster {} details:'.format(cluster_num))
        print('-' * 20)
        print('Key features:', cluster_details['key_features'])
        print('book in this cluster:')
        print(', '.join(cluster_details['books']))
        print('=' * 40)


if __name__ == '__main__':
    # 开始走起
    book_data = pd.read_csv('./data/data.csv')
    print(book_data.head())   # 查看一下前5行数据

    book_title = book_data['title'].tolist()  # 把"title"所对应的那一列转为列表
    book_content = book_data['content'].tolist()

    print("书名:", book_title[10])
    print("书的内容:", book_content[10][: 10])  # 只看第10本书内容的前10个字

    # 对内容进行处理
    norm_book_content = normalize_corpus(book_content)

    # 对内容建立特征矩阵
    vectorizer, feature_matrix = build_feature_matrix(norm_book_content,
                                                      feature_type='tfidf',
                                                      min_df=0.2,
                                                      max_df=0.9,
                                                      ngram_range=(1, 2))
    # print(feature_matrix.shape)  # 转换后的矩阵规格  (2822, 16281)
    # print(vectorizer.get_feature_names())    # 也就是将那些词作为特征
    # print("看一下前五个文本转为向量的样子:", feature_matrix[:5])
    feature_names = vectorizer.get_feature_names()

    num_clusters = 10
    km_obj, clusters = k_means(feature_matrix=feature_matrix, num_clusters=num_clusters)
    book_data['Cluster'] = clusters  # 把咱们聚类的结果放在原始数据的一列中

    # 获取每个cluster的数量
    c = Counter(clusters)
    print(c.items())

    cluster_data = get_cluster_data(clustering_obj=km_obj,
                                    book_data=book_data,
                                    feature_names=feature_names,
                                    num_clusters=num_clusters,
                                    topn_features=5)

    print_cluster_data(cluster_data)







