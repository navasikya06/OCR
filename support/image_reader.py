import sys
import os
import cv2
import numpy as np
import random
import glob
import unicodedata
import imgaug as ia
import imageio
from imgaug import augmenters as iaa
from augment import *
from PIL import Image, ImageDraw, ImageFont
import shutil

aug = iaa.SomeOf((1, 3), [
    iaa.ContrastNormalization((0.25, 1.5)),
    iaa.Affine(shear=(-2, 2)),
    iaa.Resize((0.5, 1.2)),
], random_order=True)

def aug_image(img):
    if np.random.randint(0, 10) < 4:
        return aug.augment_image(img)
    if np.random.randint(0, 10) < 4:
        return distort(img, img.shape[0] // 5)
    if np.random.randint(0, 10) < 4:
        return stretch(img, img.shape[0] // 5)
    if np.random.randint(0, 10) < 4:
        return perspective(img)
    return img

def _get_images(dir_in):
    folders = os.listdir(dir_in)
    files = []
    for folder in folders:
        #print(folders)
        for root, dirs, fs in os.walk(dir_in+folder):
            #print(fs)
            for f in fs:
                if f[-4:] in ['.jpg', '.png', 'jpeg', '.JPG']:
                    files.append(os.path.join(root, f))
    return files


def text_path(dir_in):
    image_filenames = _get_images(dir_in)
    for img_path in image_filenames:
        txt_path = '.'.join(img_path.split('.')[:-1]) + ".txt"
        print(txt_path)
        name = open(txt_path, 'w+')
        line = img_path.split('/')[-1]
        line = line.split('.')[0]
        print(line)
        name.write(line)

def _padding_image(image):
    new_h, new_w = image.shape[:2]
    h, w = image.copy().shape[:2]
    ratio = 2
    net_h, net_w = new_h//ratio, new_w//ratio

    # determine the new size of the image
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) // new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) // new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image, (new_w, new_h))
    resized = cv2.resize(resized, (w, h))

    return resized

"""
        if net_c == 1:
            resized = np.expand_dims(resized, -1)
            # embed the image into the standard letter box
            new_image = np.zeros((net_h, net_w, net_c))
        else:
            # embed the image into the standard letter box
            if mask is None:
                new_image = np.zeros((net_h, net_w, net_c))
            else:
                new_image = mask
        if is_center:
            new_image[(net_h - new_h) // 2:(net_h + new_h) // 2, (net_w - new_w) // 2:(net_w + new_w) // 2, :] = resized
        else:
            new_image[:new_h, :new_w, :] = resized
    """

    # new_image = new_image.astype(np.float32)
    # new_image = new_image/255.



def create_gif(image_list, gif_name, duration=0.1):
    frames = []
    for image in image_list:
        frames.append(image)
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def fake_background(src, dst):
    if dst.shape[1] - src.shape[1] < 5:
        return src
    if dst.shape[0] - src.shape[0] < 5:
        return src

    def padding_image(img, w_max, h_max):
        h, w = img.shape[:2]
        mask = np.ones((h_max, w_max, 3), dtype=img.dtype) * 255
        if h > h_max:
            w = max(int(w * h_max / h), 1)
            h = h_max

        if w > w_max:
            h = max(int(h * w_max / w), 1)
            w = w_max

        new_width = min(w, w_max)
        img = cv2.resize(img, (w, h))
        return img

    idx = random.randint(0, dst.shape[1] - src.shape[1] - 1)

    dst = dst[:, idx:idx + src.shape[1], :]

    idx = random.randint(0, dst.shape[0] - src.shape[0] - 1)

    dst = dst[idx:idx + src.shape[0], :, :]

    src = padding_image(src, dst.shape[1], dst.shape[0])

    src_mask = 255 * np.ones(src.shape, src.dtype)

    center = dst.shape[1] // 2, dst.shape[0] // 2

    output = cv2.seamlessClone(src, dst, src_mask, center, cv2.MIXED_CLONE)
    diff = cv2.absdiff(dst, output)

    return output

def effect(im):
    backgrounds = _get_images('paper')
    background = cv2.imread(np.random.choice(backgrounds))
    im = fake_background(im.copy(), background)
    n = random.randint(0,3)
    if n == 0:
        im = distort(im, im.shape[0] // random.randint(2,5))
    elif n == 1:
        im = stretch(im, im.shape[0] // random.randint(2,5))
    elif n == 2:
        im = perspective(im)
    return im

def effect_more(im):
    n = random.randint(0,8)
    if n == 0:
        im = noise_blur(im)
    elif n == 1:
        im = noise_pepper(im)
    elif n == 2:
        im = noise_salt(im)
    elif n == 3:
        im = transform(im)
    elif n == 4 :
        im = balance_bright(im, 100)
    elif n == 5:
        im = change_contrast(im, 100)
    elif n == 6:
        im = _padding_image(im)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #thresh = random.randint(126,129)
    #im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)[1]
    return im

"""

if __name__ == '__main__':
    #files = _get_images('/home/tchu/Desktop/result/')
    #print(len(files))
"""