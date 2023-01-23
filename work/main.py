import cv2
import numpy as np
url = r"C:\Users\17627\Pictures\furryWolf\FmZ4Z6FaAAE-XFo.jpg"
img = cv2.imread(url)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------------------Sobel边缘检测------------------------
x = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1)
# cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
# 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
Scale_absY = cv2.convertScaleAbs(y)
result = cv2.addWeighted(Scale_absX, 5, Scale_absY, 0.5, 0)
# ----------------------显示结果----------------------------
sum_img = np.hstack((Scale_absX,Scale_absY,result))
cv2.imshow('sum_img', sum_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

