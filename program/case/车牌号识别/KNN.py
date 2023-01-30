import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import cv2
import os
import numpy as np


class KNN:
    templateDict = {}
    words = []

    def read_image(self, url):
        """
        读取并预处理图像
        :return: img
        """
        # 有些图像的路径包含中文，cv2不敏感
        # img = cv2.imread(url)
        img = cv2.imdecode(np.fromfile(url), 1)

        # print(img.shape)
        # 注意如果这里有二值化的图像处理步骤，需要与普通的图像读入的方式保持一致

        gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return gary

    def cut_img(self, img):
        """
        用于拆分图像成为可以用于机器学习的数据
        :param img: img
        :return: np.array
        """
        # 调整图像为指定的大小
        img = cv2.resize(img, (20, 20))
        return img.reshape(1, 400)[0].astype(np.float32)

    def map_image(self, url='template'):
        for i, file in enumerate(os.listdir(url)):
            self.templateDict[i] = file
            self.words.append(os.listdir(url + '/' + file))
            self.words[-1] = [f'{url}/{file}/{x}' for x in self.words[-1]]  # 获取所有图像对象的路径

    def show(self, img):
        """
        用于图像展示
        :param img:img
        :return: key
        """
        cv2.imshow('img', img)
        return cv2.waitKey()

    def train(self, cut=0.8, verification=False):
        """
        训练测试对象
        :param cut: 用于作为测试集与训练集的数据比例
        :return: 返回模型的正确率
        """
        in_trains, out_trains = [[], []]  # 初始化输入参数与输出参数
        for key, words in enumerate(self.words):

            for url_img in words:
                img = self.read_image(url_img)
                out_data = self.cut_img(img)

                # 此处绘制一个输入输出对象
                in_trains.append(out_data)
                out_trains.append(key)

        ## 以下为机器学习部分

        # 打乱输入与输出的数据
        temp = [x for x in range(len(in_trains))]  # 创建一个中间变量作为下标序列
        random.shuffle(temp)
        in_trains = np.array(in_trains)
        out_trains = np.array(out_trains)

        in_trains = in_trains[temp[:]]
        out_trains = out_trains[temp[:]]

        # 划分训练集与测试集
        cut_number = int(len(in_trains) * cut)

        x_train, x_test = in_trains[:cut_number, :], in_trains[cut_number:, :]
        y_train, y_test = out_trains[:cut_number], out_trains[cut_number:]
        print("start")

        ## 交叉验证获取最优参数(可选)
        if verification:
            best_k, best_score = [0, 0]
            for k in range(5, 11):  # 外层循环搜索k
                knn = KNeighborsClassifier(weights="distance", n_neighbors=k)
                scores = cross_val_score(knn, x_train, y_train, cv=3)  # 3折交叉验证
                score = np.mean(scores)  # 当前这一组超参数在验证集上的平均分
                if score > best_score:
                    best_k, best_score = k, score
        else:
            best_k = 5
        # 训练knn模型

        self.knn = KNeighborsClassifier(weights="distance", n_neighbors=best_k)
        self.knn.fit(x_train, y_train)
        # 模型测试

        result = self.knn.predict(x_test)
        correct = np.count_nonzero(result == y_test)
        print(result, y_test)
        accuracy = correct / result.size
        print(accuracy)

        return accuracy

    def find_char(self, img=None):
        """
        图片配对
        :param img: img or url
        :return: 猜测的字符
        """
        if img is None:
            img = self.read_image(img)
        img = self.cut_img(img)
        number = self.knn.predict([img])[0]
        print(self.templateDict[number])
        return self.templateDict[number]

    def findAll(self, plateChars):
        result = ''
        for plateChar in plateChars:
            result += self.find_char(plateChar)
        return result

    def __init__(self):
        self.map_image()  # 创建图像映射关系表
        self.train()  # 获取训练模型


if __name__ == '__main__':
    response = KNN()
    response.find_char('')
