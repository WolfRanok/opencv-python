import cv2
import pyautogui
import numpy as np
from cut import cut
from sentence import sentence

"""
对目录中其他三个字符进行封装
"""


class window:
    ## 配置选型
    debug_cut = False
    debug_sentence = False
    debug_window = True
    ## 用于限制绘制屏幕截图的两个点
    begin_dot = (0, 0)
    end_dot = (950, 900)

    wait = 10  # 屏幕刷新间隔ms
    windowName = 'image'  # 窗口名称

    # 初始化两个图片处理对象
    response_cut = cut(debug=debug_cut)
    response_sentence = sentence(debug=debug_sentence)

    def __init__(self):
        pass

    def start(self):
        ## 设置窗口属性
        # 这个的作用是创建一个名为self.windowName的窗口，只要使用imshow的就可以在这个窗口中编辑
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)  # 初始化窗口名称
        cv2.moveWindow(self.windowName, self.end_dot[0], 0)  # 设置窗口左上角的位置，点的横坐标和纵坐标
        cv2.resizeWindow(self.windowName, *self.end_dot)  # 设置窗口的宽度和长度

        # 开始循环截图
        key = 0

        while key != 27:
            key = self.video()
            print(key)

    def video(self):
        """
        用于获取屏幕截图
        :return: int
        """
        img = pyautogui.screenshot(region=[*self.begin_dot, *self.end_dot])  # 分别代表：左上角坐标，宽高
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.imshow(self.windowName, img)
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

    def word_and_letter(self, img=None):
        """
        图形文字翻译
        :param img: 原图像
        :return: None
        """
        # 初始化读入图片
        if img is None:
            img = cv2.imread('test/3.jpg')

        # 将图片处理成单词序列
        sentences = self.response_sentence.get_pictures(img)

        # 将单词分离成字母
        for word_picture, word_contours in sentences:
            # print(word_contours)
            # 在原图像中标注单词位置
            if self.debug_window:
                img = cv2.rectangle(img, word_contours[0], word_contours[1], (255, 0, 0), 2)

            # 此方法返回字母图片和轮廓（不使用）
            letters_picture, letters_contour = self.response_cut.get_letter(word_picture)

            # 在原图像中标注字母位置
            if self.debug_window:
                # 将字母轮廓坐标调整到原图形坐标的位置
                for contour in letters_contour:
                    for i in range(2):
                        for j in range(2):
                            contour[i][j] += word_contours[0][j]
                    # print(contour)
                    img = cv2.rectangle(img, contour[0], contour[1], (0, 0, 255), 1)


        self.show(img)  # 展示图像


if __name__ == '__main__':
    video = window()
    video.word_and_letter()
