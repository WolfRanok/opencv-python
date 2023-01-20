import cv2
import cv2 as cv
import numpy as np
import math


class Test:
    url = r"C:\Users\17627\Desktop\2.jpg"

    def getBinary(self, img, lowest=180):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 先转灰度图像
        _, binary = cv.threshold(gray, lowest, 255, cv.THRESH_BINARY_INV)  # 转二值图

        return binary

    def open(self, binary, number=3):
        """
        开运算与闭运算
        :return:
        """
        kernel = np.ones(shape=[3, 3], dtype=np.uint8)

        dstOpen = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=number)

        return dstOpen

    def textContour(self, img=None, binary=None):
        if img is None:
            img = cv.imread(self.url)
        if binary is None:
            binary = self.getBinary(img)

            # contours 是一个有关轮廓的集合
            binary = self.open(binary, 5)

        contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 获取轮廓

        # cv.imshow('binary',binary)
        img = cv.drawContours(img, contours, -1, (0, 0, 255), 3, lineType=cv.LINE_AA)

        cv.imshow('contours', img)
        cv.waitKey()
        cv.destroyAllWindows()

    def textConvexHull(self):
        img = cv.imread(self.url)
        binary = self.getBinary(img)

        contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        resultPoint = cv.convexHull(contours[0])  # 只测试其中一个轮廓

        print(resultPoint, contours[0])

    def show(self, img):
        cv.imshow('show', img)
        cv.waitKey()

    def example_8_3(self):
        """
        构造凸包
        :return:None
        """
        # 获取二值图像
        img = cv.imread(self.url)
        img = cv.resize(img, None, fx=2, fy=2)
        binary = self.getBinary(img, 230)

        # 模糊处理
        binary = cv2.GaussianBlur(binary, (5, 5), 0, 0)

        binary = self.open(binary)  # 进行开运算
        self.show(binary)  # 展示开运算结果

        self.textContour(img, binary)  # 展示轮廓
        contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 获取轮廓

        # 凸包获取
        max_index = 0

        for i, contour in enumerate(contours):
            if cv.contourArea(contours[max_index]) > cv.contourArea(contour):
                max_index = i

        cnt = contours[0]  # 获取其中最大的轮廓
        hull = cv.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)

        # 构造图像
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]  # 分别取得第一个凸缺陷的，起点索引，终点索引，最远点索引，最远距离
            stark = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            cv.line(img, stark, end, [0, 255, 0], 2)
            cv.circle(img, far, 5, [255, 0, 0], -1)
        cv.imshow('img', img)
        cv.waitKey()
        cv.destroyAllWindows()

        # print('defects = ', defects)

    def __init__(self):
        self.example_8_3()


class gesture_recognition:
    url = r"C:\Users\17627\Desktop\0.jpg"
    cap = cv2.VideoCapture(0, cv.CAP_DSHOW)  # 这是一个摄像头对象

    left = [410, 10]
    right = [610, 210]
    resultPoint = [200, 80]

    def start(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()  # 读取摄像头图像
            frame = cv.flip(frame, 1)  # 水平翻转图像
            ## 设定一个识别区域
            high = self.right[0] - self.left[0]  # 计算宽度和高度
            wide = self.right[1] - self.left[1]

            ## 设定一个识别区域
            roi = frame[self.left[1]:self.left[1] + wide, self.left[0]:self.left[0] + high]  # 将右上角作为识别的区域
            cv.rectangle(frame, self.left, self.right, (0, 0, 255), 0)  # 将选定区域标出

            ## 皮肤检测
            hsv = cv.cvtColor(roi, cv2.COLOR_BGR2HSV)  # 色彩空间转换
            lower_skin = np.array([0, 28, 70], dtype=frame.dtype)  # 设定范围下界
            upper_skin = np.array([20, 255, 255, ], dtype=frame.dtype)  # 设定范围下界
            mask = cv2.inRange(hsv, lower_skin, upper_skin)  # 确定手势所在区域

            ## 图像预处理
            kernel = np.ones((2, 2), dtype=frame.dtype)
            mask = cv.dilate(mask, kernel, iterations=4)
            mask = cv.GaussianBlur(mask, (5, 5), 100)  # 高斯滤波

            ## 轮廓查找
            contours, h = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # 获取所有轮廓
            cnt = max(contours, key=lambda x: cv.contourArea(x))  # 获取最大的轮廓
            areaCnt = cv.contourArea(cnt)  # 轮廓的面积

            ## 提取轮廓的凸包
            hull = cv.convexHull(cnt)  # 获取凸包
            areaHull = cv.contourArea(hull)  # 凸包的面积

            ## 计算占比，占比在0.9以上的就认为是0手势
            areaRatio = areaCnt / areaHull

            ## 获取凸缺陷
            hull = cv.convexHull(cnt, returnPoints=False)  # 使用索引
            defects = cv.convexityDefects(cnt, hull)  # 获取凸缺陷

            ## 凸缺陷处理
            n = 0  # 用于定义缺陷的个数
            # 遍历手计算缺陷个数

            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                startPoint = tuple(cnt[s][0])  # 线段的起点
                endPoint = tuple(cnt[e][0])  # 线段的终点
                far = tuple(cnt[f][0])  # 最远点

                cv.line(roi, startPoint, endPoint, [0, 255, 0], 1)

                # 计算三条边长
                a = math.sqrt((endPoint[0] - startPoint[0]) ** 2 + (endPoint[1] - startPoint[1]) ** 2)
                b = math.sqrt((endPoint[0] - far[0]) ** 2 + (endPoint[1] - far[1]) ** 2)
                c = math.sqrt((far[0] - startPoint[0]) ** 2 + (far[1] - startPoint[1]) ** 2)

                # 角度计算
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                # 计算有效凸缺点的个数
                # 角度小于90且距离大于20像素才算作一个缺陷点
                if angle < 85 and d > 20:
                    n += 1
                    cv.circle(roi, far, 3, (255, 0, 0), -1)  # 用蓝色点标记有效点位置

            # 获取最终手势
            result = 0 if n == 0 and areaRatio > 0.9 or areaCnt < 2400 else n + 1

            ## 结果展示
            # 在图像上标注文字
            cv.putText(frame, f'it is {result}', self.resultPoint, cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv.imshow('frame', frame)
            k = cv.waitKey(25) & 0xff
            if k == 27:
                return

    def __init__(self):
        self.start()
        cv.destroyAllWindows()
        self.cap.release()


if __name__ == '__main__':
    gesture_recognition()
