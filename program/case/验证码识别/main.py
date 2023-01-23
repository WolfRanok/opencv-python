import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = 'd://Program Files (x86)//Tesseract-OCR//tesseract.exe'

tessdata_dir_config = '--tessdata-dir "d://Program Files (x86)//Tesseract-OCR//tessdata"'


class cracking_verification_code:
    def __init__(self):
        pass

    def show_verification(self):
        img = self.get_image()
        img = self.bigger(img, 9)

        highImg, wideImg, _ = img.shape
        boxes = pytesseract.image_to_boxes(img, lang='eng', config=tessdata_dir_config)

        for box in boxes.splitlines():
            box = box.split(' ')
            x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
            cv2.rectangle(img, (x, highImg - y), (w, highImg - h), (50, 50, 255), 2)
            cv2.putText(img, box[0], (x, highImg - y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)

        cv2.imshow('img', img)
        cv2.waitKey(0)

    def get_verification_code(self):
        img = self.get_image()
        code = pytesseract.image_to_string(img, lang='eng', config=tessdata_dir_config)
        return re.sub('\s', '', code)

    def bigger(self, img, number=3):
        return cv2.resize(img, None, fx=number, fy=number)

    def get_image(self, url='images/1.jpg'):
        """
        获取一个图片对象
        :return: img
        """
        img = cv2.imread(url)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # return img


if __name__ == '__main__':
    response = cracking_verification_code()
    print(response.get_verification_code())
    response.show_verification()
