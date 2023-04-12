import os
import cv2
import numpy as np
import itertools
import CTC as model_keras
from config import config

VOC_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints/vin.txt')
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints/vin.hdf5')

model = model_keras.cnn_ctc(training=False, input_shape=(config.INPUT_HEIGHT, config.INPUT_WIDTH, config.CHANNEL), size_voc = config.SIZE_VOC)
model.load_weights(CHECKPOINT_PATH, by_name=True, skip_mismatch=True)
model.summary()
voc = {}
data = open(VOC_PATH, 'r', encoding='utf8').readlines()

for line in data:
    k, c = line.replace('\n', '').split('\t')
    voc[int(k)] = c

def prepare_image1(img):
    img = cv2.bitwise_not(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = config.INPUT_HEIGHT / img.shape[0]
    img = cv2.resize(img, None, fx=scale, fy=scale)
    if img.shape[1] < config.INPUT_WIDTH:
        # Padding
        a = np.array([[0] * ((config.INPUT_WIDTH - img.shape[1]) // 2)] * config.INPUT_HEIGHT)
        img = np.concatenate([a, img], axis=1)
        b = np.array([[0] * (config.INPUT_WIDTH - img.shape[1])] * config.INPUT_HEIGHT)
        img = np.concatenate([img, b], axis=1)
    else:
        img = cv2.resize(img, None, fx=config.INPUT_WIDTH / img.shape[1], fy=1)
    img = img.astype(np.float32)
    img = img / 255.
    return img


def prepare_image3(img):
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
    img = img / 255.
    return img


def decode_label(out):
    out_best = list(np.argmax(out[0, :], axis=1))  # get max index -> len = 32
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    outstr = ''
    for i in out_best:
        if i < len(letters):
            outstr += letters[i]
    return outstr.replace('-', '')


def img2text(img):
    img = prepare_image3(img)
    # img = np.expand_dims(img, axis = -1)
    out = model.predict(np.array([img]))[0]
    idx = []
    for i in range(out.shape[0]):
        key = np.argmax(out[i])
        idx.append(key)
    out_best = [k for k, g in itertools.groupby(idx)]
    out_best = [voc[k] for k in out_best if k != config.SIZE_VOC - 1]
    return ''.join(out_best).replace('<null>', '')


def _get_images(dir_in):
    files = []
    for root, dirs, fs in os.walk(dir_in):
        for f in fs:
            if f[-4:] in ['.jpg', '.png', 'jpeg', '.JPG']:
                files.append(os.path.join(root, f))
    return files


def read_file_txt(path_txt):
    with open(path_txt, 'r') as f:
        lines = f.readlines()
        return lines[0].lower().replace('\n', '').strip()


if __name__ == '__main__':
    img_paths = _get_images('binary/VIN/')
    correct = 0
    for i, img_path in enumerate(img_paths):
        print(img_path)
        img = cv2.imread(img_path)
        txt = img2text(img)
        print(txt)
        label = read_file_txt(img_path.replace('.jpg', '.txt'))
        if txt == label:
            correct += 1
        # cv2.imwrite(f'debug/{txt}_{i}.jpg',img*255)
    print(f'correct: {correct}/{len(img_paths)}')