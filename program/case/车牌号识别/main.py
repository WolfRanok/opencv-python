from analysis import analysis
from cut import cut_card
from distinguish import cracking_code
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from KNN import KNN
import threading

"""
该函数为最终脚本文件，其封装了有关车牌的提取，分割，配对三个自定义类
"""


class final_distinguish_car:
    def show(self, img):
        cv2.imshow('img', img)
        cv2.waitKey()

    def __init__(self, url=r'image_car\4.jpg', debug=True):
        ## 车牌的提取
        self.analysis = analysis(url, debug=debug)
        License_plate, contour = self.analysis.start_find()
        self.show(License_plate)

        ## 车牌字符的分割
        self.cut_card = cut_card(img=License_plate, debug=debug)
        images = self.cut_card.handle_cut()

        ## 配对车牌的字符
        self.cracking_code = cracking_code()
        self.card = self.cracking_code.findAll(images)

        ## 展示结果
        img = cv2.imread(url)
        contour[1] -= 50  # 把位置留出来
        img = self.cv2ImgAddText(img, self.card, *contour, (255, 0, 0), 50)

        self.show(img)

    def cv2ImgAddText(self, img, text, left, top, textColor=(0, 255, 0), textSize=20):
        if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "simsun.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text((left, top), text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


class final_distinguish_car_machine:
    def show(self, img):
        cv2.imshow('img', img)
        cv2.waitKey()

    def init_knn(self):
        """
        初始化训练模型
        :return: None
        """
        print('开始训练')
        self.knn = KNN()
        print('训练结束')

    def __init__(self, url=r'image_car\4.jpg', debug=True):

        ## 多线程分配knn初始化计算工作
        thread_knn = threading.Thread(target=self.init_knn)
        thread_knn.start()  # 开始工作
        print("开始读取车牌信息")
        ## 车牌的提取
        self.analysis = analysis(url, debug=debug)
        License_plate, contour = self.analysis.start_find()
        self.show(License_plate)
        print('车牌已提取')

        ## 车牌字符的分割
        self.cut_card = cut_card(img=License_plate, debug=debug)
        images = self.cut_card.handle_cut()
        print('车牌字符已经分割完成')

        ## 配对车牌的字符
        thread_knn.join()   # 等待初始化线程的结束
        self.card = self.knn.findAll(images)

        ## 展示结果
        img = cv2.imread(url)
        contour[1] -= 50  # 把位置留出来
        img = self.cv2ImgAddText(img, self.card, *contour, (255, 0, 0), 50)

        self.show(img)

    def cv2ImgAddText(self, img, text, left, top, textColor=(0, 255, 0), textSize=20):
        if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "simsun.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text((left, top), text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    car = final_distinguish_car_machine(r'image_car\2.jpg',debug=True)
    # print(car.card)
