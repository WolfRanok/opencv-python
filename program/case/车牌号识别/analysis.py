import cv2

"""
本程序用于解析汽车照片并提取出完成的车牌号的照片
"""


class analysis:
    def __init__(self, url=r'image_car\4.jpg', debug=True):
        self.debug = debug
        self.img = self.get_img(url)

    def start_find(self):
        """
        查找并返回最终的结果
        :return: None
        """

        self.show(self.img)
        car_card,contour = self.handle(self.img)
        car_card = self.bigger(car_card)
        self.show(car_card)
        return car_card,contour

    @staticmethod
    def bigger(img, multiple=5):
        """
        用于图像放大
        :param img: 图片对象
        :param multiple: 图片缩放倍数
        :return: img
        """
        return cv2.resize(img, None, fx=multiple, fy=multiple)

    def show(self, img, name='img'):
        """
        图片展示
        :param img:图片对象
        :param name: 图片名称
        :return: None
        """
        if self.debug:
            cv2.imshow(name, img)
            cv2.waitKey()

    def handle(self, img):
        """
        用于图像的预处理
        :param img:BGR图片对象
        :return: None
        """
        ## 预处理，获取图片的基本信息
        copy_img = img.copy()
        area = copy_img.shape[0] * copy_img.shape[1]

        # 高斯滤波
        copy_img = cv2.GaussianBlur(copy_img, (3, 3), 0)
        self.show(copy_img)  # 展示滤波后的图像

        ## 转灰度图像
        gary = cv2.cvtColor(copy_img, cv2.COLOR_BGR2GRAY)
        self.show(gary)

        ## 边缘检测
        SobelX = cv2.Sobel(gary, cv2.CV_16S, 2, 0)  # 这里对x方向做偏导
        SobelY = cv2.Sobel(gary, cv2.CV_16S, 0, 2)  # 这里对y方向做偏导
        Sobel_img = cv2.addWeighted(SobelX, 0.5, SobelY, 1, 0)  # 教材上直接使用了SobelX作为最终结果，但是我选择使用全微分
        absX = cv2.convertScaleAbs(Sobel_img)  # 将结果映射到【0-255】的空间
        self.show(absX)

        ## 二值化处理
        ret, binary = cv2.threshold(absX, 0, 255, cv2.THRESH_OTSU)
        self.show(binary)

        ## 闭运算：先膨胀后腐蚀
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        copy_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernelX)
        self.show(copy_img)

        ## 开运算：先腐蚀后膨胀
        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
        copy_img = cv2.morphologyEx(copy_img, cv2.MORPH_OPEN, kernelX)
        self.show(copy_img)

        ## 中值滤波
        copy_img = cv2.medianBlur(copy_img, 15)
        self.show(copy_img)

        ## 轮廓处理
        contours, w1 = cv2.findContours(copy_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        copy_img = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 3)  # 将轮廓画在BGR副本上
        self.show(copy_img)

        ## 寻找车牌，取出最有可能是车牌的轮廓
        result = []
        for item in contours:
            if cv2.contourArea(item) * 10 > area:  # 过于大的部分抛弃
                continue

            rect = cv2.boundingRect(item)
            x, y, weight, height = rect
            result.append(rect)  # 收集所有的轮廓信息
            # print(weight, height, abs(weight / height - 3))

        # 提取比例最接近440/140的轮廓，该比例来源于百度
        contour = min(result, key=lambda point: abs(point[2] / point[3] - 440 / 140))
        # print(abs(contour[2] / contour[3] - 440 / 140))

        contour = [[contour[0], contour[1]], [contour[0] + contour[2], contour[1] + contour[3]]]

        copy_img = cv2.rectangle(img.copy(), contour[0], contour[1], (0, 0, 255), 3)  # 将轮廓画在BGR副本上
        self.show(copy_img)

        return img[contour[0][1]:contour[1][1], contour[0][0]:contour[1][0]],contour[0]

    @staticmethod
    def get_img(url):
        """
        用于获取函数图像
        :param url:车的图片路径
        :return:img
        """
        img = cv2.imread(url)

        return img

    @staticmethod
    def save(img_save, name='finish_img.jpg'):
        url = rf'image_car\{name}'
        cv2.imwrite(url, img_save)


if __name__ == '__main__':
    response = analysis(debug=True)
    img = response.start_find()

    cv2.imshow('response', img)
    cv2.waitKey()
    response.save(img)

    cv2.destroyAllWindows()
