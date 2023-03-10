import cv2
import numpy as np

"""
此脚本用于对含有字符的图片进行提取，注意这个脚本不需要拥有句子提取的能力
"""


class cut:
    debug = None
    min_area = None
    top = 1

    def show(self, img=None):
        """
        用于测试，并展示图像，默认展示原图像
        :param img: 用于展示的图像
        :return: Key
        """
        if not self.debug:
            return
        if img is None:
            cv2.imshow('image', self.img)
        else:
            cv2.imshow('image', img)
        key = cv2.waitKey()
        cv2.destroyAllWindows()

        save_name = rf'test\letters\{self.top}{chr(key)}.jpg'
        self.top += 1
        print(save_name)
        cv2.imwrite(save_name, img)

        return key

    def image_init(self, img=None):
        """
        该函数用于将图像灰度化，模糊化，二值化等操作得到可以处理的图像
        :param img: img
        :return: img,con
        """
        if img is None:
            img = self.img
        copy_img = img.copy()
        # self.show(img)  # 展示原始图像

        # # 高斯滤波
        # img = cv2.GaussianBlur(img, (5, 5), 0)      # 这里使用滤波可能会导致字母的粘合

        ## 灰度化处理
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # self.show(gray_img)

        ## 二值化
        _, binary = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)
        # self.show(binary)

        ## 开运算用以降噪
        # kernel = np.ones((2, 2))
        # image = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        # self.show(image)
        #
        ## 膨胀
        # image = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
        # if self.debug:
        #     self.show(image)

        ## 轮廓提取
        contours, w1 = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        show_img = cv2.drawContours(copy_img.copy(), contours, -1, (0, 0, 255), 2)
        # self.show(show_img)

        ## 提取有效轮廓
        result = []
        for item in contours:
            d = max(self.min_area,20)
            d = min(d,30)
            if cv2.contourArea(item) < d:  # 过于小的部分抛弃
                continue

            # if self.debug:
            #     print(cv2.contourArea(item))
            rect = cv2.boundingRect(item)
            x, y, weight, height = rect  # 这个是轮廓的信息
            result.append(rect)  # 收集所有的轮廓信息
        # contour 所表示的是每一个矩形轮廓的左上点和右下点的坐标
        # print(result)
        contours = [[[a[0], a[1]], [a[0] + a[2], a[1] + a[3]]] for a in result]
        if self.debug:
            show_img = copy_img.copy()
            for contour in contours:
                show_img = cv2.rectangle(show_img, contour[0], contour[1], (0, 0, 255), 1)
            # self.show(show_img)

        ## 剪切对应的轮廓，并返回对应的轮廓矩形的左上点和右下点的坐标集合、
        result_imgs = []
        for contour in contours:
            result_imgs.append(copy_img[contour[0][1]:contour[1][1], contour[0][0]:contour[1][0]])

        if self.debug:
            for img in result_imgs:
                self.show(img)

        return result_imgs, contours

    def cmd_dot(self, contour):
        """
        :param contour: 轮廓信息
        :return: boor
        """
        if cv2.contourArea(contour) > 1.5 * self.min_area:
            return False

        x,y,w,h = cv2.boundingRect(contour)
        if 0.75 < w/h < 1.5 and cv2.contourArea(contour)/(w*h) > 0.6:
            return True
        return False

    def get_letter(self, img):
        """
        用于返回截取结果的集合
        :return: zip
        """
        ## 读入所需的图像
        if img is None:
            img = cv2.imread('test/1.jpg')

        ## 定义噪音的最大面积
        self.min_area = img.shape[0] * img.shape[1] / 100
        # print(self.min_area)
        self.img = img
        # self.show()
        ## 图像预处理
        self.result_imgs, self.contours = self.image_init(self.img)  # 得到截图和轮廓矩阵

        return self.result_imgs, self.contours

    def __init__(self, debug=True):
        self.debug = debug


if __name__ == '__main__':
    img = cv2.imread(r'test\letter2.jpg')
    response = cut()
    imgs, _ = response.get_letter(img)
