import cv2
import numpy as np
from statistics import mode, median
from sklearn.externals import joblib
import pytesseract
from math import fabs
from my_find_text import find_text_main, noise_clear
import matplotlib.pyplot as plt
import math
import time
from PIL import Image


PATH_TO_IMG = r'C:\Users\Admin\PycharmProjects\PhysicsBasher\PhysicaBasher.git\img_recognition'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def image_rotator(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h//2)
    scale = 1
    if angle > 45:
        scale = w/h
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def resize_img(image, font_size=None):
    (H, W) = image.shape[:2]
    if not font_size:
        S = 490000
        ks = np.sqrt(S/(H*W))
        nH = int(H*ks)
        nW = int(W*ks)
        return cv2.resize(image, (nW, nH))
    else:
        font_size_standart = 19
        newH = int(font_size_standart*H/font_size)
        newW = int(newH*W/H)
        return cv2.resize(image, (newW, newH)), newH, newW


def number_of_pages_determination(raw_image, kernel_param):
    image = resize_img(raw_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, tresh = cv2.threshold(gray, 127, 255, 0)
    kernel = np.ones((kernel_param, 25))
    edged = cv2.erode(tresh, kernel)
    contour = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = sorted(contour, key=cv2.contourArea, reverse=True)
    page1 = False
    heights = []
    for c in contour[1:]:
        r = cv2.boundingRect(c)
        if r[2] > round(image.shape[1]/3):
            heights.append(r[3])
        if r[2] > round(image.shape[1]/2):
            page1 = True
    try:
        m = mode(heights)
    except:
        m = median(heights)
    return image, page1, m


def crop_half_img(image):
    middle = round(image.shape[1]/2)
    image1 = image[0:image.shape[0], 0:middle]
    image2 = image[0:image.shape[0], middle:image.shape[1]]
    return image1, image2


def sharpen_image(image):
    frame = cv2.GaussianBlur(image, (0,0), 3)
    sharpened = cv2.addWeighted(image, 1.5, frame, -0.5, 0)
    return sharpened


def find_numbers(image, template_name):
    template = cv2.imread(PATH_TO_IMG + template_name)
    h, w = template.shape[:2]
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.6
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    return image


def get_word_contours(image):
    char_ims = []
    coords = []
    thresh = cv2.adaptiveThreshold(image, 255, 1, 1, 11, 1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 3:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h > 8 and h < 30 and x > 3 and y > 3:
                roi = thresh[(y - 2):(y + 2 + h), (x - 2):(x + 2 + w)]
                char_ims.append(roi)
                coords.append((x,y,w,h))
    return char_ims, coords


def learner(chrims, s_list, r_list):
    print('learner started', len(chrims))
    number_of_charecters = len(chrims)
    count = 1
    if type(r_list) != list:
        r_list = r_list.reshape((1, r_list.shape[0]))
        temp_r = r_list.tolist()[0]
        print(temp_r)
    else:
        temp_r = r_list
    for chrim in chrims[500:]:
        cv2.imshow('a', chrim)
        print('{}/{}'.format(count, number_of_charecters))
        key = cv2.waitKey()
        if key == 27:
            break
        elif key != 32:
            temp_r.append(key)
            roismall = cv2.resize(chrim, (15, 10))
            smpl = roismall.reshape((1, 150))
            s_list = np.append(s_list, smpl, 0)
        count += 1
    temp_r = np.array(temp_r)
    s_list = s_list.astype('float32')
    r_list = temp_r.astype('float32')
    r_list = r_list.reshape((r_list.size, 1))
    joblib.dump(s_list, 'new_s_model.pkl')
    joblib.dump(r_list, 'new_r_model.pkl')
    return s_list, r_list


def get_num_rects(image, chrims, s_list, r_list, coordinates):
    knn = cv2.ml.KNearest_create()
    knn.train(s_list, cv2.ml.ROW_SAMPLE, r_list)
    rect_list = []
    for i, chrim in enumerate(chrims):
        roismall = cv2.resize(chrim, (15, 10))
        smpl = roismall.reshape((1, 150))
        smpl = smpl.astype('float32')
        ret, results, neighbours, dist = knn.findNearest(smpl, 3)
        # if results in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8')]:
        if results in [ord(')'), ord('.')]:
            rect_list.append(coordinates[i])
    for coord in rect_list:
         cv2.rectangle(image, (coord[0], coord[1]), (coord[0] + coord[2], coord[1] + coord[3]), (0,0,255))
    cv2.imshow('p', image)
    cv2.waitKey()
    return image, rect_list


def get_numeration_lane(rect_list):
    rect_list = sorted(rect_list, key=lambda tup: tup[0])
    i = 0
    groups = []
    cur_group = []
    curgroup = False
    while i < len(rect_list)-1:
        x = rect_list[i][0]
        next_x = rect_list[i+1][0]
        delta_x = fabs(next_x - x)
        if delta_x < (rect_list[i][2])/1:
            if curgroup:
                cur_group.append(rect_list[i])
            else:
                curgroup = True
                cur_group.append(rect_list[i])
        else:
            if curgroup:
                cur_group.append(rect_list[i])
                groups.append(cur_group[:])
                cur_group.clear()
                curgroup = False
        i += 1
    groups = sorted(groups, key=lambda gr: len(gr), reverse=True)
    return groups


def determin_main_line(image, grps):
    grp_column = []
    min_x = image.shape[0]
    for i, grp in enumerate(grps):
        grp = sorted(grp, key=lambda tpl: tpl[1])
        grp_height = fabs(grp[-1][1]-grp[0][1])
        grp_x = grp[0][0]
        grp_params = (i, grp_height, grp_x)
        grp_column.append(grp_params)
        if grp_x < min_x:
            min_x = grp_x
            min_grp = grp
    for rect in min_grp:
        cv2.rectangle(image, rect, (0, 255, 0), 1)
    return image, min_grp


def find_text_box(image, rct_grp):
    text_box_rects = []
    rct_grp = sorted(rct_grp, key=lambda tpl: tpl[1])
    i = 0
    while i < len(rct_grp):
        y = rct_grp[i][1]
        x = 1
        w = image.shape[:2][1]
        if i == len(rct_grp)-1:
            h = int(fabs(y - image.shape[:2][0]))
        else:
            h = int(fabs(y - rct_grp[i+1][1]))
        text_box_rects.append((x, y-int(h*0.01), w, h))
        i += 1
    for text_rect in text_box_rects:
        cv2.rectangle(image, text_rect, (25, 45, 46), 1)
    return image, text_box_rects


def crop_tasks(image, tsk_rct):
    task_images = []
    for rct in tsk_rct:
        task_image = image[rct[1]:rct[1]+rct[3], rct[0]:rct[0]+rct[2]]
        task_images.append(task_image)
    return task_images


def lines002(orig_image):
    def get_angle(x1, y1, x2, y2):
        vec_x = x2 - x1
        vec_y = y2 - y1
        angle_rad = vec_x/(math.sqrt(vec_x**2 + vec_y**2))
        angle = math.acos(angle_rad)
        print(angle*57.2958, vec_x, vec_y)
    image = orig_image.copy()
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    minLineLength = image.shape[1]//10
    # lines = cv2.HoughLinesP(image=edges, rho=0.02, theta=np.pi / 500, threshold=10, lines=np.array([]),
    #                         minLineLength=minLineLength, maxLineGap=100)
    #
    #
    # a, b, c = lines.shape
    # print(a)
    # for i in range(a):
    #     cv2.line(image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 1, cv2.LINE_AA)
    #     get_angle(lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3])
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image

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
    print('time to rotate {} to {} degrees '.format(time.time() - start, target_angle))

    rotated_image = image_rotator(image.copy(), target_angle)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # res = cv2.morphologyEx(rotated_image, cv2.MORPH_OPEN, kernel)
    return rotated_image


def clust_main(image):
    print('[INFO] starting clust_main')

    original = image.copy()
    cropped_text = find_text_main(original.copy())
    sharp_image = sharpen_image(cropped_text)
    cv2.imshow('k', sharp_image)
    cv2.waitKey()

    rotated_sharp_image = get_rotated_image(sharp_image)
    cv2.imshow('rotated', rotated_sharp_image)
    cv2.waitKey()

    return rotated_sharp_image