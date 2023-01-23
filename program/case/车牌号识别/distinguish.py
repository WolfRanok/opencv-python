from cut import cut_card
import cv2
import os
import numpy as np


class cracking_code:
    templateDict = {}
    templateImage = []
    words = []
    url = 'template'

    def __init__(self, debug=True):
        self.debug = debug
        ## 构建字符配对库
        for i, file in enumerate(os.listdir(self.url)):
            self.templateDict[i] = file
            self.words.append(os.listdir(self.url + '/' + file))
            self.words[-1] = [f'{self.url}/{file}/{x}' for x in self.words[-1]]

            # 初始化，所有的图片对象， 读取图片对象,因为cv2为中文不敏感所以用这种方式
            self.templateImage.append([cv2.imdecode(np.fromfile(template), 1) for template in self.words[-1]])

    def show(self, img):
        if self.debug:
            cv2.imshow('img', img)
            cv2.waitKey()

    def getMatchValue(self, templateImage, image):

        # 转灰度图
        templateImage = cv2.cvtColor(templateImage, cv2.COLOR_BGR2GRAY)

        # 转二值图
        ret, templateImage = cv2.threshold(templateImage, 0, 255, cv2.THRESH_OTSU)

        # 图像大小转换转成与image相同的大小
        templateImage = cv2.resize(templateImage, (image.shape[1], image.shape[0]))

        # 获取配对值
        result = cv2.matchTemplate(image, templateImage, cv2.TM_CCOEFF)

        return result[0][0]  # 返回配对的结果

    def best_temp(self, img):
        """
        用于寻找最配对的字符
        :return: str
        """
        # self.show(img)
        result = []
        for i, templates in enumerate(self.templateImage):  # 遍历所有的字
            cnt = 0
            for template in templates:  # 计算匹配程度的平均值
                result.append([self.getMatchValue(template, img), self.templateDict[i]])

        res = max(result, key=lambda x: x[0])  # 返回匹配度最大的组
        return res[1]

    def findAll(self, plateChars):
        result = ''
        for plateChar in plateChars:
            result += self.best_temp(plateChar)
        return result


if __name__ == '__main__':
    response = cut_card(debug=False)
    plateChars = response.handle_cut()  # 在这里得到了所有的切下来的字符
    temp = cracking_code().findAll(plateChars)
    print(temp)
