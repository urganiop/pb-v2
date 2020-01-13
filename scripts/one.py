import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier


# первый способ - найти контуры, которые максимально вероятно буквы
def find_external_contours(image):
    copy = image.copy()
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours =sorted(contours, key=cv2.contourArea, reverse=True)
    return contours


def crop_contours(image, contour):
    copy = image.copy()
    crops = []
    for c in contour:
        x, y, w, h = cv2.boundingRect(c)
        crop = copy[y:y+h, x:x+w]
        crops.append(crop)
    return crops


def select_letters(cropped_images):
    letters = []
    non_letters = []
    for cropped in cropped_images:
        cv2.destroyAllWindows()
        cv2.imshow('1', cropped)
        key = cv2.waitKey()
        if key == ord('a'):
            letters.append(cropped)
        elif key == ord('s'):
            non_letters.append(cropped)
        elif key == 27:
            break
    return letters, non_letters


def create_plots(letters, non_letters):
    X = []
    Y = []
    for letter in non_letters:
        gl = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
        x = np.ndarray.flatten(normalize(gl))
        Y.append(len(x))
        X.append(image_colorfulness(letter))
    plt.plot(X, Y, color='red', marker='o', markersize=2, linestyle="None")
    X = []
    Y = []
    for letter in letters:
        gl = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
        x = np.ndarray.flatten(gl)/255
        Y.append(len(x))
        X.append(image_colorfulness(letter))
    plt.plot(X, Y, color='green', marker='o', markersize=2, linestyle="None")
    plt.show()


def train_model(letters, non_letters):
    pass


def image_colorfulness(image):
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))

    # compute rg = R - G
    rg = np.absolute(R - G)

    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)


img = cv2.imread('mid.jpeg')

ex_cont = find_external_contours(img)
crop_imgs = crop_contours(img, ex_cont)
l, nl = select_letters(crop_imgs)
create_plots(l, nl)

