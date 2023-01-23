import cv2

"""
本脚本将实现车牌图片的分割
"""


class cut_card:
    def __init__(self, img=None, debug=True, download=False):
        """
        初始化函数，可以选择传入一个img对象
        :param img: img图片对象
        :param debug: 测试用的flag
        """
        self.img = img
        self.debug = debug
        self.downloadIs = download

    def show(self, img, name='img'):
        if self.debug:
            cv2.imshow(name, img)
            key = cv2.waitKey()

            if self.downloadIs and key == ord('s'):
                self.download(img)
            return key

    def handle_cut(self, img=None):
        """
        此方法将实现车牌号切割的主要功能
        :param img: 图片对象
        :return:
        """
        ## 图像预处理
        if img is None:
            img = self.img
        copy_img = img.copy()
        self.show(copy_img)

        ## 高斯模糊
        copy_img = cv2.GaussianBlur(copy_img, (3, 3), 0)
        self.show(copy_img)

        ## 转灰度图像
        gray = cv2.cvtColor(copy_img, cv2.COLOR_RGB2GRAY)
        self.show(gray)

        ## 二值化处理
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        self.show(binary)

        ## 膨胀处理，让可能断裂的字母重新拼接在一起
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        copy_img = cv2.dilate(binary, kernel)
        self.show(copy_img)

        ### 以上的图像预处理完毕，下面开始切割图像
        ## 获取轮廓
        contours, hierarch = cv2.findContours(copy_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hullContours = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 1)
        self.show(hullContours)  # 这里展示一下轮廓分布

        ## 筛选轮廓
        chars = []
        show_img = img.copy()
        for item in contours:
            rect = cv2.boundingRect(item)  # 返回一个包围矩形的信息
            x, y, w, h = rect
            chars.append(rect)
            cv2.rectangle(show_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        self.show(show_img)

        ## 将轮廓从左到右排序
        chars.sort(key=lambda point: point[0])

        ## 筛选可能是字的轮廓
        plateChars = []  # 存放小块的字母碎片
        for word in chars:
            if word[2] * 1.5 < word[3] < word[2] * 8 and word[2] > 20:
                x, y, w, h = word
                plateChar = binary[y:y + h, x:x + w]
                self.show(plateChar)
                plateChars.append(plateChar)
        # for card in plateChars:
        #     self.show(card)
        return plateChars

    @staticmethod
    def get_image(url=r'image_car\finish_img.jpg'):
        return cv2.imread(url)

    def download(self, img):
        cv2.imwrite('img.jpg', img)

    def __del__(self):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    response = cut_card(download=True)
    images = response.handle_cut()
