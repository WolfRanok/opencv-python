import cv2
import os
import numpy as np
import cut
import random

"""
此脚本用于数据清洗
"""


class brush:
    url_small = r'E:\python\githubWork\opencv-python\program\case\26zimu\data_param'
    url_big = r'E:\python\githubWork\opencv-python\program\case\26zimu\data_param_'
    url = r'E:\python\githubWork\opencv-python\program\case\template'
    debug = None
    cut_son = cut.cut(debug=False)

    def deskew(self, img, color=(255, 255, 255)):
        """
        用于字符倾斜校正
        :param img: img,这个原图像必须已经是灰色或者黑白图
        :param color: 表示填充色
        :return: correct-img
        """
        ## 出现备份与预处理
        # copy_img 必须以黑底白字的形式才行
        copy_img = img.copy()

        # self.show(copy_img)

        ## 用于获取图像特征
        m = cv2.moments(copy_img)
        if abs(m['mu02']) < 1e-2:  # 若倾斜度较小则不处理，直接返回原图像
            return img

        ## skew的计算
        skew = m['mu11'] / m['mu02']
        s = 20
        M = np.array([[1, skew, -0.5 * s * skew], [0, 1, 0]])

        affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
        size = (20, 20)

        # 倾斜校正, 并将填充色设置为白色
        copy_img = cv2.warpAffine(img, M, size, flags=affine_flags, borderValue=color)

        ## 倾斜校正前后的对比
        # self.show(np.hstack((img, copy_img)))

        return copy_img

    def data_get(self, img, operation, binary, is_deskew, vague):
        """
        用于创建一个指定的数据
        :param img: 原图像
        :param operation: 运算机制
        :param binary: 二值化数值
        :param is_deskew: 是否倾斜校正
        :param vague: 模糊程度
        :return: img
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = 255 - img  # 黑白转换
        img = cv2.morphologyEx(img, operation, np.ones((vague, vague)))
        _, img = cv2.threshold(img, binary, 255, cv2.THRESH_BINARY)
        if is_deskew:
            img = self.deskew(img)
        img = self.resize(img)

        return img

    def create_data(self, img):
        """
        数据生成器
        :return:img
        """
        operation = [cv2.MORPH_ERODE, cv2.MORPH_OPEN, cv2.MORPH_CLOSE]
        operation = random.choice(operation)  # 获取运算机制
        binary = random.choice(range(150, 200))  # 获取二值化的数值
        is_deskew = random.choice([False])  # 是否倾斜校正
        vague = random.choice(range(1, 2))  # 模糊的程度
        print(operation, binary, is_deskew, vague)
        return self.data_get(img, operation, binary, is_deskew, vague)

    def create_test(self):

        for x in os.listdir('test/letters'):
            old_img = cv2.imread(rf'test\letters\{x}')
            print(x)
            key = self.show(old_img)
            id = chr(key) + str(x[-5])

            for i in range(20):
                name = rf'..\template\{id}\{x[-5]}{i}.jpg'

                img = self.create_data(old_img)
                cv2.imwrite(name, img)

    def show(self, img):
        """
        用于图片展示
        :param img:img
        :return: key
        """
        if not self.debug:
            return
        cv2.imshow('img', img)
        key = cv2.waitKey()
        cv2.destroyAllWindows()
        return key

    def resize(self, img):
        """
        图像需要原先就是灰度图像
        将图片调整到标准大小20*20
        :param img: 原图像
        :return:finish_img
        """

        if img.shape[0] / img.shape[1] > 4:  # 表示长宽比太大了，不适合做拉伸操作
            ## 待补充，这里需要解决过度拉伸的问题，可能需要寻找填充的方法
            # 计算需要用于填充的图像的宽高
            high = img.shape[0]
            wide = int((img.shape[0] - img.shape[1]) / 2)

            grid = np.zeros((high, wide))  # 计算用于填充的黑色图片

            img = np.hstack((grid, img))  # 左右都做填充
            img = np.hstack((img, grid))  # 左右都做填充

            # 最后将填充的图像重新转换为20*20的格式
            img = cv2.resize(img, (20, 20))
            if self.debug:
                print("这里做了一次对称填充")
        else:
            img = cv2.resize(img, (20, 20))
        # 调整大小并返回
        return img

    def handle(self, img):
        """
        图片清洗
        :param img:img
        :return: finish_img
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = self.resize(img)      # 图像预处理
        # img = cv2.GaussianBlur(img, (3, 3), 0)
        # print(img.shape)

        _, img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
        # img = cv2.morphologyEx(img,cv2.MORPH_ERODE,np.ones((2,2)),iterations=1)
        img = 255 - img
        return img

    def delete(self):
        """
        清空所有模板库文件
        :return: None
        """
        for film in os.listdir(self.url):
            for x in os.listdir(fr'{self.url}\{film}'):
                os.remove(fr'{self.url}\{film}\{x}')

    def small_letter(self, letter='s'):

        for film in os.listdir(self.url_small):
            dir = fr'{self.url}\{letter + film[0]}'

            img = fr'{self.url_small}\{film}'
            img = cv2.imread(img)

            # img,_ = self.cut_son.get_letter(img)     # 对清洗数据也进行切割
            # img = img[0]

            try:
                img = self.handle(img)
                cv2.imwrite(fr'{dir}/{film}', img)
                # self.show(img)
            except:
                print(fr'{dir}/{film}')

    def big_letter(self, letter='d'):

        for film in os.listdir(self.url_big):
            dir = fr'{self.url}\{letter + film[0]}'

            img = fr'{self.url_big}\{film}'
            img = cv2.imread(img)

            # img,_ = self.cut_son.get_letter(img)    # 对清洗数据也进行切割
            # img = img[0]
            # self.show(img)
            try:
                img = self.handle(img)
                cv2.imwrite(fr'{dir}/{film}', img)

            except:
                print(fr'{dir}/{film}')

    def __init__(self, debug=True):
        self.debug = debug
        # url = r'E:\python\githubWork\opencv-python\program\case\26zimu\data_param\f_36.png'
        # img = cv2.imread(url)
        # img = self.handle(img)
        # self.show(img)
        self.delete()
        # self.big_letter()
        # self.small_letter()

        # img = cv2.imread(r'test\letters\M.jpg')

        self.create_test()


if __name__ == '__main__':
    brush()
