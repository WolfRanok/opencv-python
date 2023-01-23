from analysis import analysis
from cut import cut_card
from distinguish import cracking_code
import cv2

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
        License_plate = self.analysis.start_find()
        self.show(License_plate)

        ## 车牌字符的分割
        self.cut_card = cut_card(img=License_plate, debug=debug)
        images = self.cut_card.handle_cut()

        ## 配对车牌的字符
        self.cracking_code = cracking_code()
        self.card = self.cracking_code.findAll(images)


if __name__ == '__main__':
    car = final_distinguish_car()
    print(car.card)
