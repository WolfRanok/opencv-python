import cv2
import numpy as np

"""
本脚本用于将图片中的句子分离出来
"""


class cut_sentence:
    debug = None
    min_area = 900     # 句子的最小面积判定

    def show(self, img):
        """
        用于图形展示
        :param img:img
        :return: key
        """
        if not self.debug:
            return

        cv2.imshow('img', img)
        key = cv2.waitKey()
        cv2.destroyAllWindows()

        return key

    def change_color(self, img):
        """
        将图像进行预处理
        :param img:img 彩图
        :return: img 二值图
        """
        ## 原图像备份
        copy_img = img.copy()
        self.show(copy_img)

        ## 高斯模糊
        copy_img = cv2.GaussianBlur(img, (3, 3), 0)

        ## 图像灰度化
        copy_img = cv2.cvtColor(copy_img, cv2.COLOR_BGR2GRAY)
        self.show(copy_img)

        ## 图像二值化
        _, copy_img = cv2.threshold(copy_img, 115, 255, cv2.THRESH_BINARY_INV)
        self.show(copy_img)

        ## 开运算降噪
        copy_img = cv2.morphologyEx(copy_img, cv2.MORPH_OPEN, np.ones((2, 2)), iterations=2)
        self.show(copy_img)

        ## 膨胀运算
        copy_img = cv2.morphologyEx(copy_img, cv2.MORPH_DILATE, np.ones((5, 5)), iterations=4)
        self.show(copy_img)

        return copy_img

    def get_matrix(self, img):
        """
        将二值图像的所有轮廓最小涵盖矩阵提取出来
        :param img: img
        :return: matrix
        """
        dots = []
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 合理使用轮廓检测方式
        for contour in contours:
            x1, y1, w, h = cv2.boundingRect(contour)
            x2, y2 = x1 + w, y1 + h
            if w * h > self.min_area:   # 取出轮廓面积较小的地方
                dots.append([[x1, y1], [x2, y2]])

        return dots

    def draw_matrix(self, img, matrices, color=(0, 0, 255), wide=1):
        """
        用于将矩形轮廓绘制于图形中
        :param img: 矩形的主对角线上的两个点
        :param matrices: 矩形的主对角线上的两个点
        :param color: 颜色
        :param wide: 线条宽度
        :return: img
        """
        copy_img = img.copy()
        for matrix_1, matrix_2 in matrices:
            copy_img = cv2.rectangle(img, matrix_1, matrix_2, color, wide)
        return copy_img

    def get_sentences(self, img):
        """
        使用了封装好的各函数，获取切割好的句子图像，以及左下点的位置信息
        :param img: img 彩图
        :return: img，dots
        """
        binary = self.change_color(img)
        matrices = self.get_matrix(binary)

        images, dots = [], []
        for matrix_1, matrix_2 in matrices:
            image = img[matrix_1[1]:matrix_2[1], matrix_1[0]:matrix_2[0]]
            dots.append([matrix_1[1], matrix_2[0]])
            images.append(image)
            self.show(image)

        return images, dots, matrices

    def __init__(self, debug=True):
        self.debug = debug


if __name__ == '__main__':
    response = cut_sentence()
    old_img = cv2.imread('save/2.jpg')
    img = response.change_color(old_img)
    dot = response.get_matrix(img)
    old_img = response.draw_matrix(old_img, dot)
    response.show(old_img)
