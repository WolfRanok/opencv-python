import requests

"""
本脚本针对验证码图片的下载而存在
"""


class spider_verification:
    def __init__(self):
        pass

    url = 'http://readtest.cn/include/vdimgck.php???='
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:104.0) Gecko/20100101 Firefox/104.0',
        'Cookie': 'PHPSESSID=k03ljtd8458stqiunvlqklj6g2'
    }
    proxies = {
        'http': 'http://127.0.0.1:7890/',
        'https': 'https://127.0.0.1:7890/'
    }

    def download_img(self,name):
        with requests.get(self.url, headers=self.headers, proxies=self.proxies) as response:
            img = response.content
        url = f'images/{name}.jpg'
        with open(url,'wb') as f:
            f.write(img)
            # print(img)


if __name__ == '__main__':
    spider_verification().download_img(1)
