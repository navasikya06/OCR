# -*- coding:utf-8 -*-
# Author: RubanSeven

import random
import os
import cv2
import numpy as np
import imageio
from imgaug import augmenters as iaa
from warp import WarpMLS
from PIL import Image, ImageDraw, ImageFont
import sys
import glob
import unicodedata

def distort(src, segment):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut // 3
    # thresh = img_h // segment // 3
    # thresh = img_h // 5

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append([img_w - np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append([img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)])
    dst_pts.append([np.random.randint(thresh), img_h - np.random.randint(thresh)])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                        np.random.randint(thresh) - half_thresh])
        dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                        img_h + np.random.randint(thresh) - half_thresh])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def stretch(src, segment):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut * 4 // 5
    # thresh = img_h // segment // 3
    # thresh = img_h // 5

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, 0])
    dst_pts.append([img_w, 0])
    dst_pts.append([img_w, img_h])
    dst_pts.append([0, img_h])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        move = np.random.randint(thresh) - half_thresh
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + move, 0])
        dst_pts.append([cut * cut_idx + move, img_h])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def perspective(src):
    img_h, img_w = src.shape[:2]

    thresh = img_h // 2

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, np.random.randint(thresh)])
    dst_pts.append([img_w, np.random.randint(thresh)])
    dst_pts.append([img_w, img_h - np.random.randint(thresh)])
    dst_pts.append([0, img_h - np.random.randint(thresh)])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst

# def distort(src, segment):
#     img_h, img_w = src.shape[:2]
#     dst = np.zeros_like(src, dtype=np.uint8)
#
#     cut = img_w // segment
#     thresh = img_h // 8
#
#     src_pts = list()
#     # dst_pts = list()
#
#     src_pts.append([-np.random.randint(thresh), -np.random.randint(thresh)])
#     src_pts.append([-np.random.randint(thresh), img_h + np.random.randint(thresh)])
#
#     # dst_pts.append([0, 0])
#     # dst_pts.append([0, img_h])
#     dst_box = np.array([[0, 0], [0, img_h], [cut, 0], [cut, img_h]], dtype=np.float32)
#
#     half_thresh = thresh * 0.5
#
#     for cut_idx in np.arange(1, segment, 1):
#         src_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
#                         np.random.randint(thresh) - half_thresh])
#         src_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
#                         img_h + np.random.randint(thresh) - half_thresh])
#
#         # dst_pts.append([cut * i, 0])
#         # dst_pts.append([cut * i, img_h])
#
#         src_box = np.array(src_pts[-4:-2] + src_pts[-2:-1] + src_pts[-1:], dtype=np.float32)
#
#         # mat = cv2.getPerspectiveTransform(src_box, dst_box)
#         # print(mat)
#         # dst[:, cut * (cut_idx - 1):cut * cut_idx] = cv2.warpPerspective(src, mat, (cut, img_h))
#
#         mat = get_perspective_transform(dst_box, src_box)
#         dst[:, cut * (cut_idx - 1):cut * cut_idx] = warp_perspective(src, mat, (cut, img_h))
#         # print(mat)
#
#     src_pts.append([img_w + np.random.randint(thresh) - half_thresh,
#                     np.random.randint(thresh) - half_thresh])
#     src_pts.append([img_w + np.random.randint(thresh) - half_thresh,
#                     img_h + np.random.randint(thresh) - half_thresh])
#     src_box = np.array(src_pts[-4:-2] + src_pts[-2:-1] + src_pts[-1:], dtype=np.float32)
#
#     # mat = cv2.getPerspectiveTransform(src_box, dst_box)
#     # dst[:, cut * (segment - 1):] = cv2.warpPerspective(src, mat, (img_w - cut * (segment - 1), img_h))
#     mat = get_perspective_transform(dst_box, src_box)
#     dst[:, cut * (segment - 1):] = warp_perspective(src, mat, (img_w - cut * (segment - 1), img_h))
#
#     return dst

def noise_blur_lite(img):
    im = img.copy()
    noise = iaa.AdditiveGaussianNoise(scale=0.03 * 255)
    im = noise.augment_image(im)

    blurer = iaa.GaussianBlur(sigma=(0.2, 0.7))
    im = blurer.augment_image(im)
    return im


def noise_blur(img):
    ran_noise = round(random.uniform(0.07, 0.25), 2)

    # noise
    noise = iaa.AdditiveGaussianNoise(scale=0.1 * 255)
    im = noise.augment_image(img.copy())

    # motion_blur
    mo_blur = iaa.MotionBlur(k=random.randint(3, 6))
    im = mo_blur.augment_image(im)

    # gaussian blur
    blurer = iaa.GaussianBlur(sigma=(0.8, 1))
    im = blurer.augment_image(im)

    # noise
    noise = iaa.AdditiveGaussianNoise(scale=0.05 * 255)
    im = noise.augment_image(im)
    blurer = iaa.GaussianBlur(sigma=(0.5))
    im = blurer.augment_image(im)
    return im


def noise_pepper(img):
    blurer = iaa.Pepper(0.05)
    im = blurer.augment_image(img)
    return im


def noise_salt(img):
    round(random.uniform(0.01, 0.3), 1)
    blurer = iaa.Salt(0.2)
    im = blurer.augment_image(img)
    return im


def transform(img):
    yim, xim, ch = img.shape
    x = random.randint(0, 3)
    y = random.randint(0, 10)

    if y == 1:
        # bóp đầu
        pts1 = np.float32([[0 + x, 0], [xim - x, 0], [0, yim], [xim, yim]])
    elif y == 2:
        # bóp đít
        pts1 = np.float32([[0, 0], [xim, 0], [0 + x, yim], [xim - x, yim]])
    elif y == 3:
        # bóp trái
        pts1 = np.float32([[0, 0 + x], [xim, 0], [0, yim - x], [xim, yim]])
    elif y == 4:
        # bóp phải
        pts1 = np.float32([[0, 0], [xim, 0 + x], [0, yim], [xim, yim - x]])
    elif y == 5:
        # xéo phải
        pts1 = np.float32([[0, 0], [xim - x, 0], [0 + x, yim], [xim, yim]])
    else:
        # xéo trái
        pts1 = np.float32([[0 + x, 0], [xim, 0], [0, yim], [xim - x, yim]])

    pts2 = np.float32([[0, 0], [xim, 0], [0, yim], [xim, yim]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    im = cv2.warpPerspective(img, M, (xim, yim))
    #     plt.imshow(im)
    return im


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def calculate_brightness_grayth(im):
    hist, bins = np.histogram(im.ravel(), 256, [0, 256])
    pixels = sum(hist)
    brightness = scale = len(hist)

    for index in range(0, scale):
        ratio = hist[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale

def balance_bright(im, th):
    brightness = brightness_o = calculate_brightness_grayth(im)
    count = 0
    while brightness > th and count > 20:
        im = adjust_gamma(im, 0.9)
        brightness = calculate_brightness_grayth(im)
        count += 1
    count = 0
    while brightness < th and count > 20:
        im = adjust_gamma(im, 1.1)
        brightness = calculate_brightness_grayth(im)
        count += 1
    # print('brightness: ' + str(brightness_o) + ' ---> ' + str(brightness))
    return im


def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))

    def contrast(c):
        return 128 + factor * (c - 128)

    img = Image.fromarray(img)
    img.point(contrast)
    img = np.array(img)

    return img

