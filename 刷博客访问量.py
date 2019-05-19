import requests
from lxml import etree
import time
import random

def spider():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'}
    url = 'https://blog.csdn.net/shawroad88/article/list/{}?'
    proxies = {'http': 'http://223.166.247.206:9000'}

    proxies = {}
    for i in range(1, 5):  # 控制访问的页数  这里我的微博只有4页
        response = requests.get(url.format(i), proxies=proxies, headers=headers)
        selector = etree.HTML(response.text)
        time.sleep(random.randint(2, 4))  # 每次休息两秒 以防被封

        for j in range(2, 22):  # 这里可以用模糊提取，我忘了，只能遍历

            name = selector.xpath('//*[@id="mainBox"]/main/div[2]/div[{}]/h4/a/text()'.format(j))[1].replace('\n', '').strip()

            article_url = selector.xpath('//*[@id="mainBox"]/main/div[2]/div[{}]/h4/a/@href'.format(j))[0]

            print("当前正在访问博客:", name)

            res = requests.get(article_url, headers=headers)
            sec = etree.HTML(res.text)


            num = sec.xpath('//*[@id="mainBox"]/main/div[1]/div/div/div[2]/div[1]/span[2]/text()')[0]

            print("当前博客的访问量:", num)
            print()
            time.sleep(random.randint(2, 4))  # 每次休息两秒 以防被封

if __name__ == '__main__':

    spider()
