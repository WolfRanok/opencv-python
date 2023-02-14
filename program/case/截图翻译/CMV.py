import cv2
import numpy as np

"""
基于CMV时间字符识别
"""


class cmv:
    debug = None

    def __init__(self, debug=None):
        self.debug = debug

    def show(self, img):
        """
        用于显示图像情况
        :param img: img
        :return: key
        """
        if self.debug:
            return
        cv2.imshow('img', img)
        key = cv2.waitKey()
        cv2.destroyAllWindows()

        if key == ord('s'):
            cv2.imwrite('test/letter2.jpg', img)
        return key

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
        copy_img = self.change_color(copy_img)

        # self.show(copy_img)

        ## 用于获取图像特征
        m = cv2.moments(copy_img)
        if abs(m['mu02']) < 1e-2:  # 若倾斜度较小则不处理，直接返回原图像
            return img

        ## skew的计算
        skew = m['mu11'] / m['mu02']
        s = copy_img.shape[0]
        M = np.array([[1, skew, -0.5 * s * skew], [0, 1, 0]])

        affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
        size = copy_img.shape[::-1]

        # 倾斜校正, 并将填充色设置为白色
        copy_img = cv2.warpAffine(img, M, size, flags=affine_flags, borderValue=color)

        ## 倾斜校正前后的对比
        self.show(np.hstack((img, copy_img)))

        return copy_img

    def hog(self, img):
        """
        用于特征提取
        :param img:img
        :return: hist
        """
        ## 图像预处理
        copy_img = self.change_color(img)
        self.show(copy_img)
        copy_img = self.resize(copy_img)
        self.show(copy_img)

        ## 对两个维度做偏导
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)

        mag, ang = cv2.cartToPolar(gx, gy)
        bins = np.array(16 * ang / (2 * np.pi), dtype=copy_img.dtype)
        bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]

        ## 数据降维
        hits = [np.bincount(b.ravel(), m.ravel(), 16) for b, m in zip(bin_cells, mag_cells)]

        hits = np.hstack(hits)  # 将结果整合并返回
        return hits

    def change_color(self, img):
        """
        将图像灰度化二值化
        :param img: img
        :return: img
        """
        copy_img = cv2.GaussianBlur(img, (3, 3), 0)

        # 灰度化
        gray_img = cv2.cvtColor(copy_img, cv2.COLOR_RGB2GRAY)

        # 二值化
        _, binary = cv2.threshold(gray_img, 170, 255, cv2.THRESH_BINARY)

        return binary

    def resize(self, img):
        """
        将图片调整到标准大小20*20
        :param img: 原图像
        :return:finish_img
        """

        # 调整大小并返回
        return cv2.resize(img, (20, 20))


if __name__ == '__main__':
    # from sentence import sentence
    #
    # image = cv2.imread('test/2.jpg')
    #
    # for img, contour in sentence(image).get_pictures():
    #     """
    #     在sentence中已经得到了对应的img, contour的集合
    #     """
    #     print(contour)
    #     break
    img = cv2.imread('test/letter2.jpg')

    response = cmv()
    img_copy = response.deskew(img,color=(0,0,0))
    result = response.hog(img_copy)
    print(result)