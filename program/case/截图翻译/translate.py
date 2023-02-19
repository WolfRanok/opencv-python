import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from threading import Thread


def wait(func):
    """
    用于等待的修饰器
    :param func: func
    :return: func
    """
    wait_time = 0.5

    def wait_func(*args, **kwargs):
        time.sleep(wait_time)
        func(*args, **kwargs)
        time.sleep(wait_time)

    return wait_func


class translation_queue:
    number = 5  # 默认开启界面数
    queue = []  # 浏览器队列
    hh, tt = 0, 0  # 队列头，队列尾

    def add_browser(self):
        self.queue.append(spider())

    def __init__(self, number=None):
        ## 初始化线程池大小
        if number is not None:
            self.number = number

        ## 初始化5个浏览器
        children = []
        for _ in range(self.number):
            children.append(Thread(target=self.add_browser))
            children[-1].start()

        # 等待所有浏览器初始化完毕
        for i in range(self.number):
            children[i].join()

    def set_text(self, text):
        self.queue[self.tt].input(text)
        self.tt = (self.tt + 1) % self.number

    def get_text(self):
        text = self.queue[self.hh].get_translate_sentence()
        self.hh = (self.hh + 1) % self.number
        return text


class spider:
    url = 'https://fanyi.baidu.com/'
    path_ad = '//div[@class="app-guide-inner"]/div[@class="app-guide-aside"]/span'
    path_input = '//div[@class="textarea-wrap"]/textarea[@class="textarea"]'
    path_out = '//p[@class="ordinary-output target-output clearfix"]/span'

    @staticmethod
    def get_no_ui_browser():
        """
        获得一个无界面浏览器对象
        :return: Chrome
        """
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('blink-settings=imagesEnabled=false')  # 可以选择不加载图片以提升速度
        browser = webdriver.Chrome(chrome_options=chrome_options)

        return browser

    @staticmethod
    def get_browser():
        """
        获得一个有界面浏览器对象（用于展示或者调试）
        :return:  Chrome
        """

        return webdriver.Chrome()

    def click_object(self, object):
        """
        用于点击一个元素
        :param object:元素对象
        :return: None
        """
        self.browser.execute_script("arguments[0].click();", object)

    def close_ad(self):
        # 这里显示等待界面中的广告出现
        try:
            self.ad = WebDriverWait(self.browser, 1).until(EC.presence_of_element_located((By.XPATH, self.path_ad)))
        except:
            return
        self.click_object(self.ad)

    def __init__(self, debug=False):
        self.browser = self.get_browser() if debug else self.get_no_ui_browser()  # 获取一个浏览器对象
        self.browser.implicitly_wait(5)  # 设置隐式等待的时间

        self.browser.get(self.url)  # 打开界面
        self.close_ad()  # 关闭界面上的广告
        self.find_button()  # 寻找输入框

    def find_button(self):
        """
        此方法用于实现常用元素的查找
        :return: None
        """
        ## 寻找输入输出框框
        self.text_input = WebDriverWait(self.browser, 5).until(
            EC.presence_of_element_located((By.XPATH, self.path_input)))
        # self.text_out = self.browser.find_element(by=By.XPATH, value=self.path_out)

    def input(self, text):
        """
        用于给输入框输入句子
        :param text: 待翻译文本
        :return: bool
        """
        self.text_input.clear()  # 清除输入内容
        self.text_input.send_keys(text)  # 填写被翻译内容

    def get_translate_sentence(self):
        """
        获取翻译结果
        :return:翻译结果
        """
        try:
            self.text_out = WebDriverWait(self.browser, 5).until(EC.presence_of_element_located((By.XPATH, self.path_out)))
        except:
            return ''
        return self.text_out.text

    def translate_sentence(self, text, wait=0.5):
        """
        用于实现翻译的逻辑
        :param text: 需要翻译的文本
        :return: 翻译结果
        """
        self.input(text)
        time.sleep(wait)
        return self.get_translate_sentence()


if __name__ == '__main__':
    response = translation_queue()
    response.set_text('你好世界')
    response.set_text('hello world')
    response.set_text('my father is a lawyer')
    for _ in range(3):
        print(response.get_text())
