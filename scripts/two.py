import cv2
import time
import numpy as np
from PIL import Image
import pytesseract


# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thres = cv2.threshold(img, 157, 255, cv2.THRESH_TRUNC)
# img = Image.open('imgs/6.jpg')
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# text = pytesseract.image_to_string(img, lang="rus")
# print(text)

def image_rotator(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h//2)
    scale = 1
    if angle > 45:
        scale = w/h
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def get_rotated_image(image):
    print('[INFO] rotating image')
    start = time.time()
    max_peak = 0
    target_angle = 0
    peak_arr = []
    angles = [0,15,30,45,60,75,90]
    for phi in angles:
        Y1 = []
        image_array_y = image_rotator(image.copy(), phi)
        for row in image_array_y:
            Y1.append(1/sum(row))
        peak = max(Y1)
        if peak > max_peak:
            target_angle = phi
            max_peak = peak
        peak_arr.append(peak)
    for offset in [7, 3, 1]:
        angles = [target_angle-offset, target_angle, target_angle+offset]
        for phi in angles:
            Y1 = []
            image_array_y = image_rotator(image.copy(), phi)
            for row in image_array_y:
                Y1.append(1/sum(row))
            peak = max(Y1)
            if peak > max_peak:
                target_angle = phi
                max_peak = peak
            peak_arr.append(peak)
    print('[INFO] time to rotate {} to {} degrees '.format(time.time() - start, target_angle))

    rotated_image = image_rotator(image.copy(), target_angle)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # res = cv2.morphologyEx(rotated_image, cv2.MORPH_OPEN, kernel)
    return rotated_image, target_angle


# image = cv2.imread('imgs/10.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# rot = get_rotated_image(image)
# cv2.imshow('', rot)
# cv2.waitKey()