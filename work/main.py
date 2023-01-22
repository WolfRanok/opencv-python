import cv2

url = r"C:\Users\17627\Pictures\furryWolf\FmZ4Z6FaAAE-XFo.jpg"

img = cv2.imread(url)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print(img.shape)
