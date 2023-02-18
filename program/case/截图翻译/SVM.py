import random
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import joblib
import cv2
import numpy as np
import os
from cut import cut
from sentence import sentence

"""
基于CMV时间字符识别
"""


class svm_picture:
    debug = None
    machine_svm = None
    map_key = {}

    def __init__(self, debug=None):
        self.debug = debug

        # 初始化映射表
        for i, root in enumerate(os.listdir('../template')):
            self.map_key[i] = root[1]

    def show(self, img):
        """
        用于显示图像情况
        :param img: img
        :return: key
        """
        if not self.debug:
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
        s = 20
        M = np.array([[1, skew, -0.5 * s * skew], [0, 1, 0]])

        affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
        size = (20, 20)

        # 倾斜校正, 并将填充色设置为白色
        copy_img = cv2.warpAffine(img, M, size, flags=affine_flags, borderValue=color)

        ## 倾斜校正前后的对比
        # self.show(np.hstack((img, copy_img)))

        return copy_img

    def hog(self, img):
        """
        用于特征提取
        :param img:img
        :return: hist
        """
        ## 图像预处理
        copy_img = self.change_color(img)
        # self.show(copy_img)
        copy_img = self.resize(copy_img)
        # self.show(copy_img)

        result = copy_img.reshape(1, 400)[0].astype(np.float32)
        return result

    def change_color(self, img):
        """
        将图像灰度化二值化
        :param img: img
        :return: img
        """
        # copy_img = cv2.GaussianBlur(img, (3, 3), 0)

        # 灰度化
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 二值化
        _, binary = cv2.threshold(gray_img, 190  , 255, cv2.THRESH_BINARY_INV)

        # self.show(binary)
        return binary

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
            wide = int((img.shape[0] - img.shape[1])/2)

            grid = np.zeros((high, wide))  # 计算用于填充的黑色图片

            img = np.hstack((grid, img))      # 左右都做填充
            img = np.hstack((img, grid))  # 左右都做填充

            # self.show(img)

            # 最后将填充的图像重新转换为20*20的格式
            img = cv2.resize(img, (20, 20))
            # self.show(img)

        else:
            img = cv2.resize(img, (20, 20))
        # 调整大小并返回
        return img

    def get_data(self, proportion=0.2):
        """
        用于数据清洗和整理，返回测试数据和验证数据
        :param proportion: 测试集与验证集的比例
        :return: data
        """
        old_X, old_Y = [], []
        for i, root in enumerate(os.listdir('../template')):
            i = np.array(i)
            new_path = f'../template/{root}'
            for film in os.listdir(new_path)[:100]:
                img = cv2.imread(f'{new_path}/{film}')
                dot = self.hog(img)
                old_X.append(dot)
                old_Y.append(i)
        ## 随机打乱
        temp = [i for i in range(len(old_X))]
        random.shuffle(temp)

        old_X, old_Y = np.array(old_X), np.array(old_Y)
        old_X = old_X[temp]
        old_Y = old_Y[temp]

        ## 返回训练数据和测试数据
        n = int(len(old_X) * proportion)
        return old_X[n:], old_Y[n:], old_X[:n], old_Y[:n]

    def train(self):
        """
        模型训练
        :return: None
        """
        ## 数据分析与预处理
        x_train, y_train, x_test, y_test = self.get_data()

        ## 网格搜索,找到最优参数
        machine_svm = svm.SVC()

        param_grid = {'C': range(0, 50, 10)}  # 这里设置了参数的测试范围
        grid_search = GridSearchCV(machine_svm, param_grid, cv=3)  # 建立网格搜索器模型
        grid_search.fit(x_train, y_train)  # 开始搜索

        ## 创建分类器对象
        print("最优参数是 c= ", grid_search.best_params_)
        print("最优模型正确率 = ", grid_search.best_score_)
        self.machine_svm = grid_search.best_estimator_  # 获取最优模型

        ## 模型训练
        self.machine_svm.fit(x_train, y_train)

        ## 模型验证
        result = self.machine_svm.predict(x_test)
        correct = np.count_nonzero(result == y_test)
        accuracy = correct / result.size

        print("测试集正确率：", accuracy)

        ## 模型保存
        joblib.dump(self.machine_svm, 'model/svm.pkl')

    def recognize_letter(self, img):
        ## 模型初始化
        try:
            self.machine_svm = joblib.load('model/svm.pkl')
        except:
            self.train()

        ## 图像预处理
        # 倾斜度校正
        # img = self.deskew(img)
        # 图形特征提取
        hits = self.hog(img)

        ## 字母预测
        result = self.machine_svm.predict([hits])[0]
        # print(result)
        # print(hits.shape)
        # print(self.map_key[result])
        return self.map_key[result]


if __name__ == '__main__':

    img = cv2.imread('test/3.jpg')
    response = svm_picture(debug=True)
    # img = response.change_color(img)
    cut_son = cut(debug=False)
    for img,_ in sentence(debug=False).get_pictures():
        imgs,_ = cut_son.get_letter(img)
        for img in imgs:
            response.show(img)
            response.recognize_letter(img)

