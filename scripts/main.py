import cv2
import numpy as np
from PIL import Image
import pytesseract


from scripts.EAST import east_detector
from scripts.two import get_rotated_image, image_rotator

'''
    - east detector
    - crop detected
    - find rotate angle (gray input)
    - get rotated image
    - tesseract image
'''


def cropped_image(image, boxes):
    height, width = image.shape[:2]
    blank_image = np.full((height, width), 255, np.uint8)
    for (x, y, x1, y1) in boxes:
        cropped = image[y:y1, x:x1]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        # cropped = cv2.
        cropped = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 11)
        blank_image[y:y1, x:x1] = cropped
    return blank_image

def read_text(image):
    pil_im = Image.fromarray(image)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(pil_im, lang="rus")
    print(text)
    return text

def ip_main(image):
    # img = cv2.imread(image)
    img = image
    box = east_detector(img)
    cr = cropped_image(img, box)

    rot, ang = get_rotated_image(cr)
    img = image_rotator(img, ang)

    text = read_text(img)
    return text