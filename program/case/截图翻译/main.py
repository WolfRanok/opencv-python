## 以下为系统库函数
import re
import time
from threading import Thread,Lock
import cv2
import pyautogui
import numpy as np

## 以下为自定义库
from cut import cut
from sentence import sentence
from SVM import svm_picture
from PIL import Image, ImageDraw, ImageFont
from cut_sentence import cut_sentence
from translate import translation_queue

"""
对目录中其他三个字符进行封装
"""


class window:
    ## 配置选型
    # 有关debug的选项
    debug_cut = False
    debug_sentence = False
    debug_window = True
    debug_svm = True
    debug_cut_sentence = False

    # 有关绘图的选项
    is_letter = True
    color_letter = (153, 51, 250)
    line_wide_letter = 1

    is_word = True
    color_word = (128, 42, 42)
    line_wide_word = 2

    is_sentence = True
    color_sentence = (255, 0, 0)
    line_wide_sentence = 1

    test_color = (227, 23,13)  # 文字颜色
    line_wide_test = 30
    leading_space = 15      # 文字标注的反向缩进
    border_indent = 15      # 边框的缩进


    ## 用于限制绘制屏幕截图的两个点
    begin_dot = (0, 0)
    end_dot = (950, 900)

    ## 时间有关的参数
    wait = 100  # 屏幕刷新间隔ms
    wait_translate = 1  # 翻译的间隔时间
    wait_draw = 3       # 辅助线的刷新时间

    # 窗口信息
    windowName = 'image'

    ## 初始化图片处理对象
    response_cut_sentence = cut_sentence(debug=debug_cut_sentence)
    response_cut = cut(debug=debug_cut)
    response_sentence = sentence(debug=debug_sentence)

    ## 翻译对象模型
    number = 10     # 浏览器线程池大小
    baidu = translation_queue(number)

    ## 模型
    svm = svm_picture(debug=debug_svm)

    ## 全局变量
    contours_letter = []  # 字母的轮廓
    contours_word = []  # 单词的轮廓
    contours_sentence = []  # 句子的轮廓
    dots = []  # 标注信息的起始坐标
    distinguish_results = []  # 识别的最终结果
    translation_results = []  # 翻译的最终结果
    copy_contours_sentence = []     # 句子的轮廓2
    top = 0     # 序号

    ## 线程锁
    lock = Lock()

    def __init__(self):
        self.img = None
        self.old_img = None

    def start(self):
        ## 设置窗口属性
        # 这个的作用是创建一个名为self.windowName的窗口，只要使用imshow的就可以在这个窗口中编辑
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)  # 初始化窗口名称
        cv2.moveWindow(self.windowName, self.end_dot[0], 0)  # 设置窗口左上角的位置，点的横坐标和纵坐标
        cv2.resizeWindow(self.windowName, *self.end_dot)  # 设置窗口的宽度和长度

        # 启动辅助线
        self.line = Thread(target=self.guide)
        self.line.start()

        # 开始循环截图
        key = 0

        while key != 27:
            key = self.video()

    def guide(self):
        while True:
            ## 添加辅助线
            time.sleep(self.wait_draw)          # 每隔一定时间执行一次辅助线的描绘
            # 获取当前界面的情况
            img = pyautogui.screenshot(region=[self.begin_dot[0], self.begin_dot[0], self.end_dot[0], self.end_dot[1]])  # 分别代表：左上角坐标，宽高
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            self.up_date(img)

            # 填入需要翻译的句子
            self.lock.acquire()     # 上锁
            n = len(self.distinguish_results)
            for text in self.distinguish_results:
                self.baidu.set_text(text)
            self.lock.release()     # 解锁

            ## 翻译
            time.sleep(self.wait_translate)     # 每隔一定时间做一次翻译

            self.lock.acquire()     # 上锁
            self.translation_results = [self.baidu.get_text() for _ in range(n)]
            self.lock.release()     # 解锁

            # print(self.translation_results)

    def save(self,img):
        cv2.imwrite(f'save/{self.top}.jpg',img)
        self.top += 1

    def video(self):
        """
        用于获取屏幕截图
        :return: int
        """
        self.old_img = pyautogui.screenshot(region=[self.begin_dot[0],self.begin_dot[0], self.end_dot[0],self.end_dot[1]])  # 分别代表：左上角坐标，宽高
        self.old_img = cv2.cvtColor(np.asarray(self.old_img), cv2.COLOR_RGB2BGR)

        self.img = video.draw_information(self.old_img)   # 将涂鸦画到显示器中

        cv2.imshow(self.windowName, self.img)
        return cv2.waitKey(self.wait)

    def show(self, img):
        """
        图像展示
        :param img: img
        :return: Key
        """
        if not self.debug_window:
            return

        cv2.imshow('img', img)
        key = cv2.waitKey()
        cv2.destroyAllWindows()
        return key

    def up_date(self, img):
        """
        用于更新“全局变量”中的现有信息
        :return: None
        """
        ## 清除原有信息
        self.lock.acquire()     # 上锁
        self.contours_sentence = []
        self.contours_word = []
        self.contours_letter = []
        self.distinguish_results = []
        self.dots = []
        self.copy_contours_sentence = []
        self.lock.release()     # 解锁

        ## 遍历所有的句子和矩阵对角线信息
        lib = self.get_sentences(img)
        if lib:             # 如果lib中存在内容
            for sentence, dots in lib:
                begin_dot = [dots[0][0],dots[1][1]]  # 记录句子的位置
                result, _ = self.word_and_letter(sentence, dots[0])
                if result is not None:
                    # 翻译结果和翻译位置必须同时更新

                    self.lock.acquire()     # 上锁
                    self.distinguish_results.append(result)
                    self.copy_contours_sentence.append(begin_dot)
                    self.lock.release()     # 解锁

            # print(self.contours_sentence,self.contours_word)

    def draw_information(self, img):
        """
        用于将辅助线画在图像上
        :param img: img
        :return: None
        """
        # 图像备份
        copy_img = img.copy()

        self.lock.acquire()     # 上锁
        if self.is_letter:
            for d in self.contours_letter:
                dot_1, dot_2 = d[0],d[1]
                copy_img = cv2.rectangle(copy_img, dot_1, dot_2, self.color_letter)
        try:
            if self.is_word:
                for d in self.contours_word:
                    dot_1, dot_2 = d[0], d[1]
                    copy_img = cv2.rectangle(copy_img, dot_1, dot_2, self.color_word)
        except:
            pass
        if self.is_sentence:
            for dot_1, dot_2 in self.contours_sentence:
                copy_img = cv2.rectangle(copy_img, dot_1, dot_2, self.color_sentence)


        # 将翻译结果标注在图上
        for text, dot in zip(self.translation_results, self.copy_contours_sentence):
            # print("text", dot[0], dot[1], self.test_color, self.line_wide_test)
            copy_img = self.cv2ImgAddText(copy_img, text, dot[0], dot[1]-self.leading_space, self.test_color, self.line_wide_test)
        self.lock.release()     # 解锁
        return copy_img


    def get_sentences(self, img):
        """
        用于将图片中的句子提取出来,并给出可以用于文字标注的位置
        :param img: img
        :return: cut_img
        """
        images, dots, matrices = self.response_cut_sentence.get_sentences(img)
        self.contours_sentence = matrices  # 更新句子的轮廓信息
        self.dots += dots  # 更新标注位置的信息
        return zip(images, matrices)

    def word_and_letter(self, img, doc):
        """
        图形文字提取
        :param img: 原图像
        :param doc: 句子起始点坐标
        :return: None
        """
        ## 初始化读入图片和句子对象
        copy_img = img.copy()
        result = []

        ## 将图片处理成单词序列
        sentences = self.response_sentence.get_pictures(copy_img)

        ## 处理il问题的参数
        average_len, cnt, t = 0, 0, 0
        inx = []

        ## 为单词排序
        lib = list(sentences)
        if len(lib) == 0:   # 没有任何内容时直接退出即可
            return None,None
        lib.sort(key=lambda x: x[1][0][0])

        ## 记录左上顶点坐标

        self.lock.acquire()     # 上锁
        dot_x = lib[0][1][0][0]
        dot_y = max([x[1][1][1] for x in lib])
        self.lock.release()     # 解锁

        ## 将单词分离成字母
        for word_picture, word_contours in lib:
            word = []
            # print(word_contours)
            # 在原图像中标注单词位置
            if self.debug_window:
                copy_img = cv2.rectangle(copy_img, word_contours[0], word_contours[1], (255, 0, 0), 2)

            ## 更新单词轮廓信息
            self.lock.acquire()  # 上锁
            self.contours_word.append(list([x[0] + doc[0], x[1] + doc[1]] for x in word_contours))
            self.lock.release()  # 解锁

            # 此方法返回字母图片和轮廓（不使用）
            letters_picture, letters_contour = self.response_cut.get_letter(word_picture)

            # 在原图像中标注字母位置
            if self.debug_window:
                # 将字母轮廓坐标调整到原图形坐标的位置
                for contour in letters_contour:
                    for i in range(2):
                        for j in range(2):
                            contour[i][j] += word_contours[0][j]
                    copy_img = cv2.rectangle(copy_img, contour[0], contour[1], (0, 0, 255), 1)

            # 字符排序
            lib = list(zip(letters_picture, letters_contour))
            lib.sort(key=lambda x: x[1][0][0])

            # 开始识别
            for letter, contour in lib:
                # 更新字母的轮廓信息
                self.lock.acquire()  # 上锁
                self.contours_letter.append(list([x[0] + doc[0], x[1] + doc[1]] for x in contour))
                self.lock.release()  # 解锁

                # 识别字母
                letter = self.svm.recognize_letter(letter)
                word.append(letter)

                letter_len = contour[1][1] - contour[0][1]
                average_len += letter_len

                # 获取i或者l的长度
                if letter in 'iIl':
                    inx.append((cnt, letter_len))

                cnt += 1

            cnt += 1
            t += 1
            result += word + [' ']
            # 将单词加入句子
        # 处理i和L的问题
        try:
            average_len /= (cnt - t)
        except:
            return None,None
        # print(average_len, inx)
        for i, letter_len in inx:

            if letter_len <= average_len:  # 说明是i
                result[i] = 'i'
            else:
                result[i] = 'l'

        result = ''.join(result)
        result = result.lower()

        # 描述最终结果
        # if self.debug_window:
            # print(result)
            # copy_img = self.cv2ImgAddText(copy_img, '识别结果：' + result, dot_x, dot_y, (255, 0, 0), 35)
            # self.show(copy_img)  # 展示图像

        return result, (dot_x, dot_y)

    def cv2ImgAddText(self, img, text, left, top, textColor=(0, 255, 0), textSize=20):
        """
        用于为图片显示之后中文的函数
        :param img: 原图像
        :param text: 文字信息
        :param left: 顶点坐标的横坐标
        :param top: 顶点坐标的纵坐标
        :param textColor: 文本颜色
        :param textSize: 字体大小
        :return:img
        """
        if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)

        # 字体的格式
        fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")

        # 绘制文本
        draw.text((left, top), text, textColor, font=fontStyle)

        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    video = window()
    # img = cv2.imread('test/2.jpg')
    # # img = video.cv2ImgAddText(img,'text', 23, 92 ,(0, 255, 255), 40)
    # video.up_date(img)
    # img = video.draw_information(img)
    # video.show(img)
    video.start()
