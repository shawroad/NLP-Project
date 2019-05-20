"""

@file   : 001-实现逆向最大匹配.py

@author : xiaolu

@time1  : 2019-05-07

"""

class IMM(object):
    def __init__(self, dic_path):
        self.dictionary = set()
        self.maximum = 0

        # 读取词典
        with open(dic_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                self.dictionary.add(line)
                if len(line) > self.maximum:
                    self.maximum = len(line)   # 保留的是哪个词最长(字最多)

    def cut(self, text):
        result = []
        index = len(text)
        while index > 0:
            word = None
            for size in range(self.maximum, 0, -1):   # 先匹配最长
                if index - size < 0:    # 你这句画都没有人家单词长
                    continue
                piece = text[(index-size): index]   # 从后面截取最长单词的长度
                if piece in self.dictionary:
                    word = piece
                    result.append(word)
                    index -= size    # index减一个size 为下一次做铺垫
                    break
            if word is None:  # 如果在词表中不存在， 让最大长度减一  试试看能匹配吗
                index -= 1
        return result[::-1]  # 将列表逆向输出 就是我们的分词结果

if __name__ == '__main__':
    text = "南京市长江大桥"
    tokenizer = IMM('./data/imm_dic.utf8')
    # 上面文件中是词表  内容如:南京市  南京市长 长江大桥 人名解放军 大桥
    result = tokenizer.cut(text)
    print(result)
