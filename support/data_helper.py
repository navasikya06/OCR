import cv2, sys, os, timeit
import numpy as np
from keras.applications.vgg16 import preprocess_input
from image_reader import _padding_image, _get_images, aug_image
from config import config
from keras.utils import Sequence

def get_voc(dir_in = ''):
    if os.path.exists(config.VOC_PATH):
        voc = {}
        data = open(config.VOC_PATH,'r',encoding ='utf8').readlines()
        size_voc=len(data)
        for line in data:
            k,c = line.replace('\n','').split(' ')
            # matrix = [0]*size_voc
            # matrix[int(k)]=1
            voc[c]=int(k)
        return voc
"""
    voc = set()
    max_size_text = 0
    image_filenames = _get_images(dir_in)
    for img_path in image_filenames:
        txt_path = '.'.join(img_path.split('.')[:-1]) + ".txt"
        if not os.path.exists(txt_path):
            os.remove(img_path)
            continue
        print(txt_path)
        name = open(txt_path, 'r', encoding='utf8').readlines()
        name = ''.join(name).strip()
        max_size_text = max(max_size_text, len(name))
        for c in name:
            if '\\' in c:
                os.remove(img_path)
                os.remove(txt_path)
                break
            if '+' in c:
                os.remove(img_path)
                os.remove(txt_path)
                break
            if '_' in c:
                os.remove(img_path)
                os.remove(txt_path)
                break
            if '-' in c:
                os.remove(img_path)
                os.remove(txt_path)
                break
            if '*' in c:
                os.remove(img_path)
                os.remove(txt_path)
                break

            voc.add(c)
    voc.add('<null>')

    fw = open(config.VOC_PATH, 'w', encoding ='utf-8')
    idx = 0
    for key in voc:
        fw.write(f'{idx}\t{key}\n')
        idx +=1 
    fw.close()

    print('max size text:',max_size_text)
    print('size voc:',idx+1)
    print('files',len(image_filenames))
    return voc"""

# def compare(voc1_path,voc2_path):
#     def load(voc_path):
#         if os.path.exists(voc_path):
#             voc = set()
#             data =open(voc_path,'r',encoding ='utf8').readlines()
#             size_voc=len(data)
#             print(size_voc)
#             for line in data:
#                 k,c = line.replace('\n','').split('\t')
#                 voc.add(c)
#             return voc

#     voc1 = load(voc1_path)
#     voc2 = load(voc2_path)
#     return voc1.difference(voc2)

def get_data_val():
    def load(voc_path):
        if os.path.exists(voc_path):
            voc = set()
            data =open(voc_path,'r',encoding ='utf8').readlines()
            size_voc=len(data)
            print(size_voc)
            for line in data:
                k,c = line.replace('\n','').split('\t')
                voc.add(c)
            return voc
    def is_checked(name,data):
        for c in name: 
            if c in data:
                return True
        return False
    voc = load(config.VOC_PATH)
    image_filenames = _get_images('data')
    key_file ={}
    for img_path in image_filenames:
        txt_path = '.'.join(img_path.split('.')[:-1]) + ".txt"
        name =open(txt_path,'r',encoding ='utf8').readlines()
        name = ''.join(name).strip()
        for c in name:
            if '\\' in c: continue


def slide_image(image,windows=[],step=4):
        h, w = image.shape[:2]
        output_image = []
        half_of_max_window = max(windows) // 2
        for center_axis in range(half_of_max_window, w - half_of_max_window, step):
            slice_channel = []
            for window_size in windows:
                image_slice = image[:, center_axis - window_size // 2: center_axis + window_size // 2,:]
                image_slice = cv2.resize(image_slice, (32, 32))
                slice_channel.append(image_slice)
            output_image.append(np.dstack(slice_channel))
        return np.asarray(output_image, dtype='float32')


# def prepare_images(img_path):
#     img = cv2.imread(img_path)
#     img = cv2.bitwise_not(img)
#     img =  cv2.resize(img, (config.INPUT_WIDTH, config.INPUT_HEIGHT))
#     img = img.astype(np.float32)
#     img = img/255.
#     return img

def prepare_images(img_path):
    img = cv2.imread(img_path)
    img = aug_image(img)
    # img = cv2.bitwise_not(img)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    scale = config.INPUT_HEIGHT / img.shape[0]
    img = cv2.resize(img, None, fx=scale, fy=scale)
    if img.shape[1] < config.INPUT_WIDTH:
        # Padding
        a = np.array([[0] * ((config.INPUT_WIDTH - img.shape[1]) // 2)] * config.INPUT_HEIGHT)
        a = np.array([a] * 3)
        a = np.resize(a, (a.shape[1], a.shape[2], a.shape[0]))
        img = np.concatenate([a, img], axis=1)

        b = np.array([[0] * (config.INPUT_WIDTH - img.shape[1])] * config.INPUT_HEIGHT)
        b = np.array([b] * 3)
        b = np.resize(b, (b.shape[1], b.shape[2], b.shape[0]))
        img = np.concatenate([img, b], axis=1)
    else:
        img = cv2.resize(img, None, fx=config.INPUT_WIDTH / img.shape[1], fy=1)
    
    img = img.astype(np.float32)
    img = img/255.
    return img

class MY_Generator(Sequence):

    def __init__(self, dir_in, batch_size, input_shape):

        self.image_filenames = _get_images(dir_in)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.voc = get_voc()
    
    def __len__(self):
        return len(self.image_filenames) // self.batch_size

    def __getitem__(self, idx):
        x_batch = []
        y_batch = []
        label_length = np.zeros((self.batch_size, 1))
        input_length = np.zeros([self.batch_size, 1])

        img_paths = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        # if len(img_paths)<self.batch_size:
        #     print(idx,'err',self.__len__())
        #     exit()
        for i, img_path in enumerate(img_paths):
            img_path = str(img_path)
            txt_path = '.'.join(img_path.split('.')[:-1]) + ".txt"
            name = open(txt_path,'r',encoding ='utf8').readlines()
            name = ''.join(name).strip()
            pad = (config.MAX_SIZE_TEXT - len(name))*['<null>']
            label = []
            for c in name:
                label.append(self.voc[c])
            for c in pad:
                label.append(self.voc[c])

            # print('image shape', img.shape)
            img = np.expand_dims(img, axis = -1)
            img = slide_image(img,config.WINDOWS,config.CHARACTER_STEP)
            x_batch.append(img)
            y_batch.append(label)
            # print(label)
            label_length[i] = len(name)
            input_length[i] = img.shape[1]//8
            # print('input_length',label_length[i])
            # print('label_length',input_length[i])
        
        # print('batch_size shape', np.array(x_batch).shape)
        inputs = {
            'the_input': np.array(x_batch),
            'the_labels': np.array(y_batch),
            'input_length': input_length,
            'label_length': label_length
        }
        outputs = {'ctc': np.zeros([self.batch_size])}
        # print('the_input',inputs['the_input'].shape)
        # print('the_labels',inputs['the_labels'].shape)
        # print('input_length',inputs['input_length'].shape)
        # print('label_length',inputs['label_length'].shape)
        # print('ctc',outputs['ctc'].shape)
        # print(inputs['the_labels'].shape)
        return inputs, outputs

if __name__=='__main__':
    voc = get_voc("checkpoints/voc.txt")
    print(voc)