import cv2
import pyautogui
import numpy as np

"""
对目录中其他三个字符进行封装
"""


class window:
    # 用于限制绘制屏幕截图的两个点
    begin_dot = (0, 0)
    end_dot = (950, 900)

    wait = 10  # 屏幕刷新间隔ms
    windowName = 'image'  # 窗口名称

    def __init__(self):
        ## 设置窗口属性
        # 这个的作用是创建一个名为self.windowName的窗口，只要使用imshow的就可以在这个窗口中编辑
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)     # 初始化窗口名称
        cv2.moveWindow(self.windowName, self.end_dot[0], 0)     # 设置窗口左上角的位置，点的横坐标和纵坐标
        cv2.resizeWindow(self.windowName, *self.end_dot)        # 设置窗口的宽度和长度

        # 开始循环截图
        self.start()

    def start(self):
        key = 0

        while key != 27:
            key = self.video()
            print(key)

    def video(self):
        """
        用于获取屏幕截图
        :return: int
        """
        img = pyautogui.screenshot(region=[*self.begin_dot, *self.end_dot])  # 分别代表：左上角坐标，宽高
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.imshow(self.windowName, img)
        return cv2.waitKey(self.wait)


if __name__ == '__main__':
    window()
