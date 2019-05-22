"""

@file   : 005-识别人名，地名等NER.py

@author : xiaolu

@time1  : 2019-05-06

"""
from pyhanlp import *


def select_name():
    # 提取一段话中的中国人名
    text = '张三下午去王五家玩耍，中途看到了张华，然后张三就爱上张华了。'
    segment = HanLP.newSegment().enableNameRecognize(True)
    result = []
    for term in segment.seg(text):
        if str(term.nature) == 'nr':
            result.append(term.word)
    result = set(result)
    print("这一句话中所含的人名如下: \n" + ' '.join(result))


def select_address():
    # 提取下面一句中的地址

    text = '我爱北京。北京市中国的首都。西安是世界十大古都，爱中国，爱北京，爱西安，爱兵马俑旁边的韩庄村。'
    segment = HanLP.newSegment().enablePlaceRecognize(True)
    result = []
    for term in segment.seg(text):
        if str(term.nature) == 'ns':
            result.append(term.word)
    result = set(result)
    print("这一句话中所含的地址名如下: \n" + ' '.join(result))


def select_organize():

    # 组织名的识别
    text = '上海世贸组织是哪一年成立的？ 中国电子科技集团录用我作为他们公司的董事。'
    segment = HanLP.newSegment().enableOrganizationRecognize(True)
    result = []
    for term in segment.seg(text):
        if str(term.nature) == 'ntc':
            result.append(term.word)
    result = set(result)
    print("这一句话中所含的地址名如下: \n" + ' '.join(result))


if __name__ == '__main__':
    # 命名实体识别(NER) 之提取人名
    select_name()

    # 提取地址名
    select_address()

    # 提取组织名
    select_organize()
