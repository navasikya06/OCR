#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFilter
import cv2
import numpy as np
import random
from im_proc import *
from imgaug import augmenters as iaa
import time
import sys 
# from split_image import *
import json
import xlwt
# import xlsxwriter

numbers = '123456789'
letters = 'ABCDEFGHJKLMNPRSTUVWXYZ1234567890'
letters0 = 'ABCDEFGHJKLMNPRSTUVWXYZ'
letters1 = 'ABCDEFGHJKLMNPRSTUVWXYZ123456789ABCDEFGHJKLMNPRSTUVWXYZABCDEFGHJKLMNPRSTUVWXYZ'
# RL4 BH9FK 6 C 6 000196
WMI = ['AHT', 'JT', 'LTV', 'MBJ', 'MHF', 'MR0', 'NMT', 'SB1', 'RL4', 'LFM', 'LVG', '7A4', 
'RL4', 'TW1', 'VNK', '2T', '4T', '5T', '6T1', '8AJ', '93R', '9BR', 'PN1']

def make_folder_from_path(path):
    keys = path.split("/")
    passed = '/' if path.startswith('/') else ''
    for key in keys:
        passed = os.path.join(passed, key)
        if passed.strip() == "" or passed.strip() == "/":
            continue
        if not os.path.exists(passed):
            os.makedirs(passed)

def gen_VIN():
    ran = random.randint(0,40)
    ranT = 'T' if random.randint(0,9) < 7 else random.choice(letters)
    v1 = ''
    if ran < 8:
        v1 = random.choice(WMI)
    elif ran < 9:
        v1 = 'R' + random.choice('LMNPR') + ranT
    elif ran < 10:
        v1 = random.choice(letters0) + random.choice(letters) + ranT
    elif ran < 11: # US
        v1 = random.choice(numbers) + ranT + random.choice(letters)
    if len(v1) <=2:
        v1 += random.choice(letters)
    #spec
    v2 = ''
    for i in range(2):
        v2 += random.choice(letters)
    v2 += random.choice(letters + '0')
    for i in range(2):
        v2 += random.choice(letters)
    #val code
    v3 = random.choice(letters)
    #year
    v4 = random.choice(['H', 'J', 'K', 'L', 'M', 'N'])
    #factory
    v5 = random.choice(letters1)
    #serie
    serie = ''
    for i in range(6):
        serie += str(random.randint(0, 9))

    # print(v1 + '_' + v2 + '_' + v3 + '_' + v4 + '_' + v5 + '_' + serie)
    # print(len(v1 + v2 + v3 + v4 + v5 + serie))
    return v1 + v2 + v3 + v4 + v5 + serie

def gen_EIN():
    serie = ''
    for i in range(4):
        serie += random.choice(letters)
    for i in range(4):
        serie += str(random.randint(0, 9))
    serie += 'TR'
    return serie

def gen_plate():
    plate = str(random.randint(11, 99))
    plate += random.choice(letters0)
    plate += '-'
    for i in range(3):
        plate += str(random.randint(0, 9))
    plate += '.'
    for i in range(2):
        plate += str(random.randint(0, 9))
    return plate


def put_text_cv2(im, loc, txt, size, color):
    font_name = ImageFont.truetype('fonts/times.ttf', size)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    draw.text(loc, txt, color, font=font_name)
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    return im


def gen_A4():
     for i in range(20):
        sheet = 'b%d_p.'%i

        im = Image.open('abc.jpg')
        draw = ImageDraw.Draw(im)
        font_name = ImageFont.truetype('times.ttf', 50)
        font_name_bd = ImageFont.truetype('timesbd.ttf', 50)
        f = open(sheet + 'txt', 'w')
        draw.text((800, 2240), sheet, 'black', font=font_name)
        xl = 110+50
        xr = 860+50
        y = 90-10
        lx = xl
        font = font_name
        txt = 'Số máy: ' + gen_EIN()
        f.write(txt.split(':')[-1].strip() + '\n')
        for character in txt:
            if character == ':':
                font = font_name_bd
            draw.text((lx, y), character, 'black', font=font)
            w_char, h_char = font_name.getsize(character)
            lx += w_char

        lx = xr
        font = font_name
        txt = 'Số máy: ' + gen_EIN()
        f.write(txt.split(':')[-1].strip() + '\n')
        for character in txt:
            if character == ':':
                font = font_name_bd
            draw.text((lx, y), character, 'black', font=font)
            w_char, h_char = font_name.getsize(character)
            lx += w_char

        for i in range(18):
            y+=114
            lx = xl
            font = font_name
            txt = 'Số máy: ' + gen_EIN()
            f.write(txt.split(':')[-1].strip() + '\n')
            for character in txt:
                if character == ':':
                    font = font_name_bd
                draw.text((lx, y), character, 'black', font=font)
                w_char, h_char = font_name.getsize(character)
                lx += w_char

            lx = xr
            font = font_name
            txt = 'Số máy: ' + gen_EIN()
            f.write(txt.split(':')[-1].strip() + '\n')
            for character in txt:
                if character == ':':
                    font = font_name_bd
                draw.text((lx, y), character, 'black', font=font)
                w_char, h_char = font_name.getsize(character)
                lx += w_char

        im = np.asarray(im)
        
        print(im.shape)
        cv2.imwrite(sheet + 'jpg', im)

def balance_bright(im, th):
    brightness = brightness_o = calculate_brightness_grayth(im)
    count = 0
    while brightness > th and count > 20:
        im = adjust_gamma(im,0.8)
        brightness = calculate_brightness_grayth(im)
        count += 1
    count = 0
    while brightness < th and count > 20:
        im = adjust_gamma(im,1.5)
        brightness = calculate_brightness_grayth(im)
        count += 1
    # print('brightness: ' + str(brightness_o) + ' ---> ' + str(brightness))
    return im

def img2bin(img):
    
    # print(calculate_brightness_grayth(img))
    im = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    im = cv2.medianBlur(im, 5)
    im = cv2.GaussianBlur(im, (5, 5), 0)
    im = balance_bright(im, 0.7) #75
    im = increase_contrast(im)
    im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 4)
    return im

#=======================================================================================
if __name__ == '__main__':
    gen_A4()
