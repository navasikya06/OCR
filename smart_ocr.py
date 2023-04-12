import os
import cv2
import sys
import random
import timeit
import numpy as np
import xlsxwriter
#debug
from PIL import Image, ImageDraw, ImageFont

#EAST
from modules_so.east.make_line import *
from modules_so.east.text_detector import east_seg

#YOLO
from modules_so.yolo.seg import get_text_box, get_char_box


#ATTENTION
from modules_so.attention_ocr.predict import add_ocr#, code_ocr

#CNN
from modules_so.cnn.predict import predict_char, predict_num
# =================================================================================
debug = 1
RESULT = []
FILE_NAME = ''

def valid_img(img):
    try:
        img.shape
        return True
    except:
        return False

def overlap_box(boxA, boxB):
    #boxA : EAST
    #boxB : YOLO
    boxA = list(map(int, boxA))
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    #combine boxes
    # new_box = [boxB[0], boxA[1], boxB[2], boxA[3]]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou



def padding_image(img, w_max=48, h_max=32):
    h, w = img.shape[:2]
    mask = np.ones((h_max, w_max,3), dtype=img.dtype)*255
    if h > h_max:
        w = max(int(w * h_max / h),1)
        h = h_max
    if w > w_max:
        h = max(int(h * w_max / w),1)
        w = w_max
    img = cv2.resize(img, (w, h))
    return img

def resize_image(im, min_side_len=1024):
    h, w, _ = im.shape

    ratio = min_side_len/min(h,w)

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    im = cv2.resize(im, (resize_w, resize_h))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, ratio_h

def put_text_cv2(im, loc, txt, size, color):
    font_name = ImageFont.truetype('libs_so/times.ttf', size)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    draw.text(loc, txt, color, font=font_name)
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    return im
# a51G985914 - 51G-985.91 - 51G-985.911
def correct_plate_simple(plate):
    if plate.isdigit():
        return plate
    elif len(plate) >= 10:
        plate = plate[1:-1]
    elif len(plate) >= 9:
        if not plate[3].isdigit():
            plate = plate[1:]
        elif not plate[2].isdigit():
            plate = plate[:-1]
    plate = plate[:3] + '-' + plate[3:]
    plate = plate[:-2] + '.' + plate[-2:]
    return plate

def relu(x):
    return max(0, x)

# main ==========
def img2txt(im):
    # im, ratio = resize_image(im, 512)
    im1 = im
    h,w = im.shape[:2]

    e1= timeit.default_timer()
    merged_boxes, group_mask = east_seg.get_merged_boxes_old(im)
    if merged_boxes == None:
        return('Null')
    # merged_boxes = east_seg.get_merged_boxes(im)
    print(f'EAST-time: {round(timeit.default_timer()-e1,2)}')

    # ============================ YOLO FIELDS ============================
    cboxes = get_text_box(im)
    cboxes = sorted(cboxes, key=lambda x: x.xmax)
    cboxes[:2] = sorted(cboxes[:2], key=lambda x: x.ymin)

    # if len(cboxes) > 2:
    #     plate = [max(0, cboxes[0].xmin - int(cboxes[0].xmin * 0.025)), max(0, cboxes[0].ymin - int(cboxes[0].ymin * 0.015)), cboxes[0].xmax + int(cboxes[0].xmax * 0.02), cboxes[0].ymax + int(cboxes[0].ymax * 0.015)]
    #     EIN   = [max(0, cboxes[1].xmin - int(cboxes[0].xmin * 0.015)), max(0, cboxes[1].ymin - int(cboxes[1].ymin * 0.015)), cboxes[1].xmax + int(cboxes[1].xmax * 0.02), cboxes[1].ymax + int(cboxes[1].ymax * 0.015)]
    #     VIN   = [max(0, cboxes[2].xmin - int(cboxes[0].xmin * 0.015)), max(0, cboxes[2].ymin - int(cboxes[2].ymin * 0.02)), cboxes[2].xmax + int(cboxes[2].xmax * 0.02), cboxes[2].ymax + int(cboxes[2].ymax * 0.02)]
    if len(cboxes) > 2:
        plate = [max(0, cboxes[0].xmin - int(cboxes[0].xmin * 0.005)), max(0, cboxes[0].ymin - int(cboxes[0].ymin * 0.015)), cboxes[0].xmax + int(cboxes[0].xmax * 0.015), cboxes[0].ymax + int(cboxes[0].ymax * 0.015)]
        EIN   = [max(0, cboxes[1].xmin - int(cboxes[0].xmin * 0.005)), max(0, cboxes[1].ymin - int(cboxes[1].ymin * 0.015)), cboxes[1].xmax + int(cboxes[1].xmax * 0.015), cboxes[1].ymax + int(cboxes[1].ymax * 0.015)]
        VIN   = [max(0, cboxes[2].xmin - int(cboxes[0].xmin * 0.015)), max(0, cboxes[2].ymin - int(cboxes[2].ymin * 0.02)), cboxes[2].xmax + int(cboxes[2].xmax * 0.015), cboxes[2].ymax + int(cboxes[2].ymax * 0.02)]
    else:
        plate = [0, 0, 10, 10]
        EIN = [0, 0, 10, 10]
        VIN = [0, 0, 10, 10]
    croped_plate = im[plate[1]:plate[3], plate[0]:plate[2]]
    croped_EIN = im[EIN[1]:EIN[3], EIN[0]:EIN[2]]
    croped_VIN = im[VIN[1]:VIN[3], VIN[0]:VIN[2]]
    
    # ============================ YOLO CHAR ============================
    cboxes_plate = get_char_box(croped_plate)
    cboxes_EIN = get_char_box(croped_EIN)
    cboxes_VIN = get_char_box(croped_VIN)
    #sort boxes
    cboxes_plate = sorted(cboxes_plate, key=lambda x: x.xmin)
    cboxes_EIN = sorted(cboxes_EIN, key=lambda x: x.xmin)
    cboxes_VIN = sorted(cboxes_VIN, key=lambda x: x.xmin)

    croped_plate_debug = croped_plate
    croped_EIN_debug = croped_EIN
    croped_VIN_debug = croped_VIN

    chars = []
    digits = []
    ocr_plate = []
    ocr_EIN = []
    ocr_VIN = []
    # adjust and crop chars img
    if len(cboxes_plate) >= 5:
        for char in cboxes_plate:
            if char.xmax - char.xmin > 5 and char.ymax-char.ymin > 5:
                pad_x = int((char.xmax - char.xmin) * 0.05)
                pad_y = int((char.ymax - char.ymin) * 0.05)
                
                if char.label == 0:
                    cv2.rectangle(croped_plate_debug,(max(char.xmin-pad_x, 0), max(char.ymin-pad_y, 0)), (min(char.xmax + pad_x, croped_plate.shape[1]), min(char.ymax + pad_y, croped_plate.shape[0])),(0,0,255),1)
                    chars.append(croped_plate[max(0, char.ymin - pad_y):min(croped_plate.shape[0], char.ymax + pad_y), max(0,char.xmin - pad_x):min(croped_plate.shape[1], char.xmax + pad_x)])
                    ocr_plate += '0'
                if char.label == 1:
                    cv2.rectangle(croped_plate_debug,(max(char.xmin-pad_x, 0), max(char.ymin-pad_y, 0)), (min(char.xmax + pad_x, croped_plate.shape[1]), min(char.ymax + pad_y, croped_plate.shape[0])),(0,255,0),1)
                    digits.append(croped_plate[max(0, char.ymin - pad_y):min(croped_plate.shape[0], char.ymax + pad_y), max(0,char.xmin - pad_x):min(croped_plate.shape[1], char.xmax + pad_x)])
                    ocr_plate += '1'
    if len(cboxes_EIN) >= 5:
        for char in cboxes_EIN:
            if char.xmax - char.xmin > 5 and char.ymax-char.ymin >5:
                pad_x = int((char.xmax - char.xmin) * 0.05)
                pad_y = int((char.ymax - char.ymin) * 0.05)
                if char.label == 0:
                    cv2.rectangle(croped_EIN_debug,(max(char.xmin-pad_x, 0), max(char.ymin-pad_y, 0)), (min(char.xmax + pad_x, croped_EIN.shape[1]), min(char.ymax + pad_y, croped_EIN.shape[0])),(0,0,255),1)
                    chars.append(croped_EIN[max(0, char.ymin - pad_y):min(croped_EIN.shape[0], char.ymax + pad_y), max(0,char.xmin - pad_x):min(croped_EIN.shape[1], char.xmax + pad_x)])
                    ocr_EIN += '0'
                if char.label == 1:
                    cv2.rectangle(croped_EIN_debug,(max(char.xmin-pad_x, 0), max(char.ymin-pad_y, 0)), (min(char.xmax + pad_x, croped_EIN.shape[1]), min(char.ymax + pad_y, croped_EIN.shape[0])),(0,255,0),1)
                    digits.append(croped_EIN[max(0, char.ymin - pad_y):min(croped_EIN.shape[0], char.ymax + pad_y), max(0,char.xmin - pad_x):min(croped_EIN.shape[1], char.xmax + pad_x)])
                    ocr_EIN += '1'
    if len(cboxes_VIN) >= 5:
        for char in cboxes_VIN:
            if char.xmax - char.xmin > 5 and char.ymax-char.ymin >5:
                pad_x = int((char.xmax - char.xmin) * 0.05)
                pad_y = int((char.ymax - char.ymin) * 0.05)
                if char.label == 0:
                    cv2.rectangle(croped_VIN_debug,(max(char.xmin-pad_x, 0), max(char.ymin-pad_y, 0)), (min(char.xmax + pad_x, croped_VIN.shape[1]), min(char.ymax + pad_y, croped_VIN.shape[0])),(0,0,255),1)
                    chars.append(croped_VIN[max(0, char.ymin - pad_y):min(croped_VIN.shape[0], char.ymax + pad_y), max(0,char.xmin - pad_x):min(croped_VIN.shape[1], char.xmax + pad_x)])
                    ocr_VIN += '0'
                if char.label == 1:
                    cv2.rectangle(croped_VIN_debug,(max(char.xmin-pad_x, 0), max(char.ymin-pad_y, 0)), (min(char.xmax + pad_x, croped_VIN.shape[1]), min(char.ymax + pad_y, croped_VIN.shape[0])),(0,255,0),1)
                    digits.append(croped_VIN[max(0, char.ymin - pad_y):min(croped_VIN.shape[0], char.ymax + pad_y), max(0,char.xmin - pad_x):min(croped_VIN.shape[1], char.xmax + pad_x)])
                    ocr_VIN += '1'

    
    # ============================ CNN predict char ============================
    digits = predict_num(digits)
    chars = predict_char(chars)

    for i, char in enumerate(ocr_plate):
        if char == '0':
            ocr_plate[i] = chars.pop(0)
        else:
            ocr_plate[i] = digits.pop(0)
    ocr_plate = ''.join(ocr_plate)
    # print(ocr_plate)


    for i, char in enumerate(ocr_EIN):
        if char == '0':
            ocr_EIN[i] = chars.pop(0)
        else:
            ocr_EIN[i] = digits.pop(0)
    ocr_EIN = ''.join(ocr_EIN)
    # print(ocr_EIN)


    for i, char in enumerate(ocr_VIN):
        if char == '0':
            ocr_VIN[i] = chars.pop(0)
        else:
            ocr_VIN[i] = digits.pop(0)
    ocr_VIN = ''.join(ocr_VIN)
    # print(ocr_VIN)
    
    ocr_plate = correct_plate_simple(ocr_plate)
    if len(ocr_VIN) >= 18: ocr_VIN = ocr_VIN[:-1]
    

    # ============================ EAST + CRNN EAST ============================
    average_h = 0
    im_box = []
    imgs = []
    predicts = []
    e2= timeit.default_timer()
    for j, line in enumerate(sorted(merged_boxes, key=lambda x:x[0][0][1])):
        for i, box in enumerate(sorted(line, key=lambda x:x[0][0])):
            # emilate boxes smaller than 10px 
            if box[2][0] - box[0][0] < 10 or box[2][1] - box[0][1] < 10:
                continue
            east_box = [min(box[0][0], box[3][0]), min(box[0][1], box[1][1]), max(box[1][0], box[2][0]), max(box[2][1], box[3][1])]
            if overlap_box(east_box, plate) > 0.1: continue
            if overlap_box(east_box, EIN) > 0.1: continue
            if overlap_box(east_box, VIN) > 0.1: continue
            # Add to predicts
            croped_im = east_crop(im,box)
            if not valid_img(croped_im): continue
            if len(imgs) == add_ocr.batch_size:
                predicts += add_ocr.img2text(imgs)
                imgs =[]

            imgs.append(croped_im)
            box = box.astype(np.int32)
            im_box.append([[np.amin(box[:,:-1]), np.amin(box[:,1:]), np.amax(box[:,:-1]), np.amax(box[:,1:])],j])
            # Canculate average_h and boxes
            average_h += np.amax(box[:,1:]) - np.amin(box[:,1:])
    average_h = average_h/len(im_box)

    # Predict the rest of set boxes
    if imgs:
        num_pad =add_ocr.batch_size - len(imgs)
        none_im = [np.zeros((1,1,3),dtype = imgs[0].dtype) for _ in range(num_pad)]
        imgs += none_im
        predicts += add_ocr.img2text(imgs)[:-num_pad]
        del imgs, none_im



# =========================== MAKE LINES =================================

    line_cur = 0
    text = []
    x_min = w + 1
    sens = []
    len_space = average_h*0.5
    num_space=0
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    for i,(box, line) in enumerate(im_box):
        if i >= len(predicts)-1: break
        if line > line_cur:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # distance
            distance = int(box[0] - im_box[i-1][0][2])
            if distance > len_space*3 and len(sens) > 0:
                # print(sens)
                sens[-1] = sens[-1] + ''.join(' ' * int(distance/len_space))

            num_space = int(x_min//len_space)
            add_space = (num_space - len(sens[-1])) if sens else 0

            if add_space > 0:
                sens[-1] = sens[-1][:-1] + ''.join([' ']*(add_space))+' '.join(text)+'\n'
            else:
                text = ''.join([' ']*(num_space))+' '.join(text)+'\n'
                sens.append(text)
            text = []
            x_min = w + 1
            line_cur = line


        # text.append(predicts[i] if 'soát' not in predicts[i] else 'soát')
        text.append(predicts[i])
        x_min = min(x_min,box[0])
        if debug:
            cv2.rectangle(im1,(box[0],box[1]),(box[2],box[3]),color,2)

    text = ''.join([' ']*(num_space))+' '.join(text)+'\n'
    sens.append(text)

# =========================== Replace ===========================
    for i, se in enumerate(sens):
        if 'Biển kiểm' in se:
            sens[i] = sens[i].replace('Biển kiểm', 'Biển kiểm soát: ' + ocr_plate)
        elif 'soát' in se.lower():
            sens[i] = sens[i].replace('soát', 'soát: ' + ocr_plate)
            sens[i] = sens[i].replace('Soát', 'soát: ' + ocr_plate)

        if 'số máy' in se.lower() or 'số khung' in se.lower() or 'soát' in se.lower():
            sens[i] = sens[i].replace('Số máy', 'Số máy: ' + ocr_EIN)
            sens[i] = sens[i].replace('số máy', 'Số máy: ' + ocr_EIN)
            sens[i] = sens[i].replace('Số khung', 'Số khung: ' + ocr_VIN)
            sens[i] = sens[i].replace('số khung', 'Số khung: ' + ocr_VIN)
        if 'QUYẾT ĐỊNH' in se:
            sens[i] = sens[i].replace('QUYẾT ĐỊNH', '\n' + ''.join([' ']*50) + 'QUYẾT ĐỊNH')




    #DEBUG=================
    if debug:
        cv2.imwrite('plate.jpg', croped_plate_debug)
        cv2.imwrite('ein.jpg', croped_EIN_debug)
        cv2.imwrite('vin.jpg', croped_VIN_debug)
        # Draw CODES
        cv2.rectangle(im1,(plate[0], plate[1]), (plate[2], plate[3]),(0,0,255),2)
        cv2.rectangle(im1,(EIN[0], EIN[1]), (EIN[2], EIN[3]),(0,0,255),2)
        cv2.rectangle(im1,(VIN[0], VIN[1]), (VIN[2], VIN[3]),(0,0,255),2)

        im1 = put_text_cv2(im1, (plate[2], plate[1]), ocr_plate, 30, (255, 0, 0))
        im1 = put_text_cv2(im1, (EIN[2], EIN[1]), ocr_EIN, 30, (255, 0, 0))
        im1 = put_text_cv2(im1, (VIN[2], VIN[1]), ocr_VIN, 30, (255, 0, 0))
        im1 = put_text_cv2(im1, (10,10), ocr_plate + ' - ' + ocr_EIN + ' - ' + ocr_VIN, 30, (255, 0, 0))
        link_save = 'debug/' + FILE_NAME[:-4]
        cv2.imwrite(link_save + '.jpg', im1)

        # statistics 
        global RESULT
        RESULT += [FILE_NAME + '_' + ocr_plate + '_' + ocr_EIN + '_' + ocr_VIN]
        # txt file
        with open(link_save + '.txt', 'w') as file:
            for s in sens:
                file.write(s+'\n')
            file.write(ocr_plate+'\n')
            file.write(ocr_EIN+'\n')
            file.write(ocr_VIN+'\n')

    print(f'OCR-time: {round(timeit.default_timer()-e2,2)}')
    print(f'total-time: {round(timeit.default_timer()-e1,2)}')
    print(f'total-lines: {len(sens)}')
    print('---')
    return sens



def write_summary_xlsx(ls_total, name_file):
    title = ["name", "plate", "ein", "vin"]
    wb = xlsxwriter.Workbook(name_file)
    sheet1 = wb.add_worksheet('sheet 01')

    sheet1.write(0,0, title[0])
    sheet1.write(0,1, title[1])
    sheet1.write(0,2, title[2])
    sheet1.write(0,3, title[3])
    row = 1
    for line in ls_total:    
        arr = line.split('_')    
        sheet1.write(row ,0, arr[0])
        sheet1.write(row ,1, arr[1])
        sheet1.write(row ,2, arr[2])
        sheet1.write(row ,3, arr[3])
        row += 1
    wb.close()



if __name__ == '__main__':
    url = sys.argv[1]
    format_file = ['.jpg', '.png', 'jpeg', 'gif']
    path_imgs = []
    if url[-4:].lower() in format_file:
        path_imgs = [url]
    else:
        for root,dirs, files in os.walk(sys.argv[1]):
            for file in files:
                if file[-4:].lower() in format_file:
                    path_imgs.append(os.path.join(root, file))



    for file in path_imgs:
        print(file)
        try:
            im = cv2.imread(file)
            print(im.shape)
            FILE_NAME = file.split('/')[-1]
            rs = img2txt(im)
            print(rs)
            write_summary_xlsx(RESULT, 'result_char2.xlsx')
        except Exception as ex:
            print(ex)

