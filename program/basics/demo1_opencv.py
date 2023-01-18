import cv2
import numpy as np


class openCv:
    url = r"C:\Users\17627\Pictures\furryWolf\FmZ4Z6FaAAE-XFo.jpg"
    picture = cv2.imread(url)

    def example_3_2(self):
        """
        最基础的图像展示
        :return: None
        """
        img = cv2.imread(self.url)  # 读取图片

        cv2.imshow("window1", img)  # 展示图像，并为窗口取名“window1”
        cv2.imshow("window2", img)  # 展示图像，并为窗口取名“window2”
        print(type(img))
        # waitKey 传入一个等待时间ms，当在等待时间中按下按键则返回按键的ASCLL，否则超时返回-1，参数默认为0，表示一直等待
        key = cv2.waitKey()
        print(key)
        cv2.destroyAllWindows()  # 关闭所有窗口

    def example_3_5(self):
        """
        灰色图像写入一个白色矩形
        :return: None
        """
        img = cv2.imread(self.url, cv2.IMREAD_GRAYSCALE)
        print(img)
        img[:20, :50] = 255

        cv2.imshow('白色矩形', img)

        cv2.waitKey()

    def example_3_7(self):
        """
        通道合并
        彩色图像按照三个信道切分为三个图像
        :return:None
        """
        img = cv2.imread(self.url)

        Blue = img[:, :, 0]  # 获取蓝色信道的图像信息
        Green = img[:, :, 1]  # 获取绿色信道的图像信息
        Red = img[:, :, 2]  # 获取红色信道的图像信息

        cv2.imshow("Old", img)
        cv2.imshow("Blue", Blue)
        cv2.imshow("Green", Green)
        cv2.imshow("Red", Red)

        img[:, :, 0] = 0
        cv2.imshow("1", img)
        img[:, :, 1] = 0
        cv2.imshow("2", img)
        img[:, :, 2] = 0
        cv2.imshow("3", img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def example_3_8(self):
        """
        通道合并
        :return:None
        """
        img = cv2.imread(self.url)
        b, g, r = cv2.split(img)

        new_img1 = cv2.merge([b, g, r])
        new_img2 = cv2.merge([r, g, b])
        cv2.imshow('new_img1', new_img1)
        cv2.imshow('new_img2', new_img2)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def example_3_9(self):
        """
        图像大小调整
        :return: None
        """
        oldImg = cv2.imread(self.url)
        smallerImg = cv2.resize(oldImg, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
        biggerImg = cv2.resize(oldImg, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("smaller", smallerImg)
        cv2.imshow("bidder", biggerImg)
        print(biggerImg.shape, type(biggerImg.shape))
        cv2.waitKey()
        cv2.destroyAllWindows()

    def example_3_12(self):
        """
        掩模（mask）的使用
        :return:None
        """
        img = cv2.imread(self.url, 1)
        # shape 为图像的大小参数元组,dtype是图像的参数类型
        mask = np.zeros(img.shape, dtype=img.dtype)

        mask[50:200, 50:300] = 1  # 掩模的制作
        mask[200:300, 20:200] = 1

        cv2.imshow("old_img", img)
        cv2.imshow("mask", mask * 255)
        cv2.imshow("new_img", img * mask)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def example_3_15(self):
        """
        提取某些色彩的图像部分
        以皮肤划分为例
        将以下返回定义为皮肤
        色调：[0 - 33]
        饱和度：[10 - 255]
        明度：[80 - 255]
        :return: None
        """
        img = cv2.imread(self.url)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        min_HSV = np.array([0, 10, 80], dtype=hsv.dtype)
        max_HSV = np.array([33, 255, 255], dtype=hsv.dtype)
        mask = cv2.inRange(hsv, min_HSV, max_HSV)
        result = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("mask", mask)
        cv2.imshow("result", result)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def example_3_17(self):
        img = cv2.imread(self.url)
        cv2.imshow('img', img)

        # 均值滤波
        blur = cv2.blur(img, (5, 5))
        cv2.imshow('blur', blur)

        # 高斯滤波
        GaussianBlur = cv2.GaussianBlur(img, (5, 5), 0, 0)
        cv2.imshow('GaussianBlur', GaussianBlur)

        # 中值滤波
        MedianBlur = cv2.medianBlur(img, 3)  # 这个3表示滤波核的高度和宽度为3，与上面两种情况不同的是这个核必须是长宽为正奇数正方形
        cv2.imshow('MedianBlur', MedianBlur)

        cv2.waitKey()
        cv2.destroyAllWindows()

    def __init__(self):
        self.example_3_17()


if __name__ == '__main__':
    openCv()
