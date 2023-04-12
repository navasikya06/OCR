#
# Authors: PHAN VAN HUNG, NGUYEN THANH DO, DAO BAO LINH, MINH PAUL
# Project: ID OCR
# GMO-Z.com RUNSYSTEM JSC
#
from builtins import list

import cv2, os, collections, re
import numpy as np
import cv_algorithms
from PIL import Image
import unicodedata
from collections import Counter
# from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance as ndld


def skeletonization(gray, background='black'):
    mask = cv_algorithms.zhang_suen(gray) if background.lower() == 'black' else cv_algorithms.zhang_suen(
        cv2.bitwise_not(gray))
    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    return mask


def make_folder_from_path(path):
    keys = path.split("/")
    passed = '/' if path.startswith('/') else ''
    for key in keys:
        passed = os.path.join(passed, key)
        if passed.strip() == "" or passed.strip() == "/":
            continue
        if not os.path.exists(passed):
            os.makedirs(passed)
            # print("created: {}".format(passed))

# ====================================================
# 
#                       CONTOURS       
# 
# ====================================================


def sort_contours(cnts, method="left-to-right"):
    if len(cnts) == 0:
        return (cnts, [])
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def find_boundingBox(im):
    _, contours, hierarchy = cv2.findContours(im,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnt, boudingBox_sorted = sort_contours(contours)

    # Removenoise
    im, boxes = remove_noise(im, boudingBox_sorted)

    # Merging contours
    # boxes = merge_boudingBox(boxes)

    # boxes = split_boudingbox_super(boxes)
    
    return im, boxes

def askMerge(cur, follow):
    x, y, w, h = cur
    x1, y1, w1, h1 = follow

    if (x + w ) - x1 > 5:
        return True
    x_ = min(x, x1)
    y_ = min(y, y1)
    far = max(x + w, x1 + w1) - x_
    t = max(y + h, y1 + h1) - y_
    if (x1 < x + w) and (x1 + w1 < x + w):
        return True
    if (x1 < (x + w)) and (t/far > 1.1):
        return True
    return False

def doMerge(a, b):
    x, y, w, h = a
    x1, y1, w1, h1 = b

    x_ = min(x, x1)
    y_ = min(y, y1)
    lb = max(x + w, x1 + w1) - x_
    t = max(y + h, y1 + h1) - y_

    temp = [x_, y_, lb, t]
    return temp


def merge_boudingBox(cnts):
    cnt = []
    for i in cnts:
        # if i[2] * i[3] > 70 and (i[2] > 3) and (i[3] > 3) and (i[2] < 255):
        temp = [i[0], i[1], i[2], i[3]]
        cnt.extend([temp])

    i=0
    while i < len(cnt):
        j = i+1
        while j < len(cnt):
            cur = cnt[i]
            follow = cnt[j]
            if askMerge(cur, follow):
                cnt[i] = doMerge(cur, follow)
                cnt.remove(follow)
                j-=1
            j += 1
        i += 1
    return cnt

def merge_boudingBox_word(boudingBox_sorted):
    box = boudingBox_sorted
    h_max = np.amax([x[3] for x in box])
    i=0
    while i < len(box) - 1:
        j = i+1
        while j < len(box):
            cur = box[i]
            follow = box[j]            
            if (follow[0] - (cur[0] + cur[2]))/h_max < 0.13:
                box[i] = doMerge(cur, follow)
                box.remove(follow)
                j-=1
            j+=1
        i+=1
    return box

def split_boudingbox(cnts):
    new_cnts = []

    for c in cnts:
        x, y, w, h = c
        if w / h >= 1.4 and w > 50:
            c1 = [[x, y, round(w/2) - 1, h]]
            c2 = [[x + round(w/2) + 1, y, round(w/2), h]]
            new_cnts.extend(c1)
            new_cnts.extend(c2)
        else:
            new_cnts.extend([c])

    return new_cnts

def split_boudingbox_super(cnts):
    new_cnts = []

    for c in cnts:
        x, y, w, h = c
        # print(str(c) + ' - ' + str(w/h))
        if w / h >= 1.15 and w > 50:
            if w / h >= 2:
                divide = round((w*1.13)/h)
                add = round(w/divide)
                for j in range(divide):
                    ci = [[x + (add*j) + 1, y, add - 1, h]]
                    new_cnts.extend(ci)
            else:
                c1 = [[x, y, round(w/2) - 1, h]]
                c2 = [[x + round(w/2) + 1, y, round(w/2) - 1, h]]
                new_cnts.extend(c1)
                new_cnts.extend(c2)
        else:
            new_cnts.extend([c])

    return new_cnts




def clean_dots(bbxs, h_max):

    err_idxs=[]
    th = 1
    if h_max <= 12:
        return bbxs
    
    if h_max > 200:
        th = 300 #260
    elif h_max > 100:
        th = 150 #50
    elif h_max > 50:
        th = 50  #4
    elif h_max > 20:
        th = 20;

    new_bbxs = []
    for i,(x, y, w, h) in enumerate(bbxs):
        if w * h < th:
            err_idxs.append([x, y, w, h])
            continue

        new_bbxs.append([x, y, w, h])
    return new_bbxs, err_idxs


def clean_bolder(bbxs, h_max):
    err_idxs = []
    new_bbxs = []

    th2 = h_max / 5
    th3 = 4 / 5 * h_max
    for i, (x, y, w, h) in enumerate(bbxs):
        if y + h < th2 or y > th3:
            err_idxs.append([x, y, w, h])
            continue
        new_bbxs.append([x, y, w, h])
    return new_bbxs, err_idxs

def remove_noise(img_bin, boxes):
    im = img_bin.copy()
    new_bbxs, err_idxs = clean_dots(boxes, im.shape[0])
    for x, y, w, h in err_idxs:
        im[y:y+h, x:x+w] = img = np.zeros((h,w))
    return im, new_bbxs

def find_noise(bbxs, h_max, gray, color_bias):
    # bbxs = clean_bolder(bbxs, h_max)[0]
    bbxs = clean_dots(bbxs, h_max)[0]
    new_bbxs = []
    for x, y, w, h in bbxs:
        if np.amax(cv2.bitwise_not(gray[y:y+h, x:x+w])) < color_bias:
            continue
        new_bbxs.append([x, y, w, h])
    return new_bbxs

def draw_all_contours(im, cnt):
    imcp = im.copy()

    for c in cnt:
        x, y, w, h, = c
        cv2.rectangle(imcp,(x,y),(x+w,y+h),(100,125,125),2)
    return imcp

def crop_img(img, cnt):
    x, y, w, h = cnt
    return img.copy()[y:y + h, x:x + w]




# ====================================================
# 
#                        CORRECTING       
# 
# ====================================================

def correcting(dic, tens):
    lstCorr = []
    for i in dic:
        lstCorr.append([i, ndld(i.upper(), tens.upper())])
    lstCorr = sorted(lstCorr, key=lambda c: c[1])
    return lstCorr[0][0]





# ====================================================
# 
#                  IMGAGE PROCCESSING       
# 
# ====================================================

def pre_im2bin(im):
    im = balance_bright(im, 0.65)
    im = remove_background(im)
    im = img2bin(im)
    return im

def img2bin(img):
    # print(calculate_brightness_grayth(img))
    im = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    im = cv2.medianBlur(im, 5)
    im = cv2.GaussianBlur(im, (5, 5), 0)
    im = balance_bright(im, 0.75)
    im = increase_contrast(im)
    im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 4)
    return im

def remove_background(im):
    image = im.copy()
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    brown_lo=np.array([0,0,0])
    brown_hi=np.array([255,150,150])
    mask=cv2.inRange(hsv,brown_lo,brown_hi)
    image[mask>0] = [50,50,50]
    mask = cv2.bitwise_not(mask)
    image[mask>0] = [200,200,200]
    return image

def balance_bright(im, th):
    brightness = brightness_o = calculate_brightness_grayth(im)
    while brightness > th:
        im = adjust_gamma(im,0.9)
        brightness = calculate_brightness_grayth(im)
    while brightness < th:
        im = adjust_gamma(im,1.1)
        brightness = calculate_brightness_grayth(im)
    # print('brightness: ' + str(brightness_o) + ' ---> ' + str(brightness))
    return im

def padding_image(self, gray, w_max=64, h_max=64):
        h, w = gray.shape[:2]
        mask = np.zeros((h_max, w_max), dtype=gray.dtype)

        if w > w_max:
            h = int(h * w_max / w)
            w = w_max
        if h > h_max:
            w = int(w * h_max / h)
            h = h_max

        gray = cv2.resize(gray, (w, h))
        x1 = max((h_max - h) // 2, 0)
        x2 = max((w_max - w) // 2, 0)

        mask[x1:x1 + h, x2:x2 + w] = gray
        # # debug:
        # idx = len(os.listdir('asset/debug'))
        # cv2.imwrite('asset/debug/'+str(idx) +'.jpg', mask)
        return mask









def gray2bin(img):
    '''
    :param img: image background must be white [shape=(?,?,1)]
    :return: img [shape=(?,?,1)]
    '''
    img = cv2.medianBlur(img, 3)
    img = cv2.GaussianBlur(img.copy(), (3, 3), 0)
    bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,12)

    return bin

def get_background(gray):
    '''

    :param gray: shape=(?,?,1)
    :return: black or white
    '''
    white = len(gray[np.where(gray > 200)])
    black = len(gray[np.where(gray < 50)])
    return 'black' if white < black else 'white'


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def convert_img(img_RGB):
    gray = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
    _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    background = get_background(gray)
    if background == 'white':
        img = cv2.bitwise_not(img_RGB)
    return img




def get_space(bbxs):
    size = len(bbxs)
    if size < 1:
        return None
    spaces = np.array([x - (bbxs[i][0] + bbxs[i][2]) for i, (x, _, _, _) in enumerate(bbxs[1:-1])])
    width = np.array([w for _, _, w, _ in bbxs[:-1]])
    height = np.array([h for _, _, _, h in bbxs[:-1]])
    space_count = collections.Counter(spaces).most_common(3)
    width = sorted(width, reverse=True)
    height = sorted(height, reverse=True)

    if size > 5:
        height = height[:5]
        width = width[:5]
    ave_height = np.average(height)
    ave_width = np.average(width)
    spaces = 0
    size = 0
    for key, number in space_count:
        spaces += key * number
        size += number

    if size == 0:
        return None

    return spaces / size, [max(ave_width, ave_height*0.9), ave_height * 0.85]


def renew_image(bin_co,gray, color_bias):
    '''
    input: binary white background image
    '''
    im_mask = np.ones_like(bin_co) * 255
    h_mask, w_mask = im_mask.shape

    _, contours, hierarchy = cv2.findContours(cv2.bitwise_not(bin_co), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, bbxs = sort_contours(contours, method='left-to-right')
    bbxs = remove_noise(bbxs, h_mask, gray, color_bias)
    for x, y, w, h in bbxs:
        im_mask[y:y + h, x:x + w] = bin_co[y:y + h, x:x + w]
    return im_mask


def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))

    def contrast(c):
        return 128 + factor * (c - 128)

    return img.point(contrast)


def increase_contrast(im):
    pil_im = Image.fromarray(im)
    return np.array(change_contrast(pil_im, 100))



def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale

def calculate_brightness_grayth(im):
    hist,bins = np.histogram(im.ravel(),256,[0,256])
    pixels = sum(hist)
    brightness = scale = len(hist)

    for index in range(0, scale):
        ratio = hist[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale

#
def remove_shadow(img):
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, diff_img.copy(), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result, result_norm


