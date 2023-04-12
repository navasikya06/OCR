import h5py as h5
import cv2
import numpy as np
import os,sys, glob
import scipy.io as spio
import random, jaconv, numpy as np
import time
import datetime
# import Draw_image as drw
import xml.etree.ElementTree as ET
import unicodedata
import time
from  mat4py import loadmat 
from colorama import Fore, Style
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from copy import deepcopy
from absl import app, flags
from tqdm import tqdm

DATASET = 'data/dset.h5'
BASELINE = 'data/dset_baselines.h5'

FLAGS = flags.FLAGS
flags.DEFINE_string('corpus', '' ,'Path to corpus')
flags.DEFINE_string('fonts',  '' ,'Path to fonts ')
flags.DEFINE_integer('num_samples', 50000, 'Number of SynthText data to create')
flags.DEFINE_integer('max_samples', 5000 , 'Number of SynthText data per folder')
flags.DEFINE_boolean('viz', False , 'Visualize result')	

H = 80
W = 600

numbers = '123456789'
letters = 'ABCDEFGHJKLMNPRSTUVWXYZ1234567890'
letters0 = 'ABCDEFGHJKLMNPRSTUVWXYZ'

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

def timing(fn):
	def wrap_timing(*args, **kwargs):
		stime = time.time()
		x = fn(*args, **kwargs)
		ptime = time.time() - stime
		print(f'{Fore.RED}Processing time {ptime} s{Style.RESET_ALL}')
		return x 
	return wrap_timing

def add(img_path, seg, depth, area, label):


	img = cv2.imread(img_path)

	filename = img_path.split('/')[-1]
	with h5.File(DATASET, 'a') as ds:
		print(ds.keys())
		depth_set = ds['depth']
		image_set = ds['image']
		seg_set = ds['seg']

		depth_set.create_dataset(filename, np.shape(depth), np.float32, data = depth)
		image_set.create_dataset(filename, np.shape(img), np.uint8, data = img)
		seg_set.create_dataset(filename, np.shape(seg), np.uint16, data = seg)
		seg_set[filename].attrs['area'] = area
		seg_set[filename].attrs['label'] =  label

def fake_map(path):

	img_list = glob.glob(path + '/*.jpg')

	depth_baseline = []
	segmap_baseline = [] 
	area_baseline = [] 
	label_baseline = []

	indx = None

	with h5.File(BASELINE, 'r') as baseline:

		image_baselines =  list(baseline['depth'].keys())
		for img in image_baselines:
			depth_baseline.append(baseline['depth'][img][...])
			segmap_baseline.append(baseline['seg'][img][...])
			area_baseline.append(baseline['seg'][img].attrs['area'])
			label_baseline.append(baseline['seg'][img].attrs['label'])

		assert len(depth_baseline) == len(segmap_baseline), "Number of Depthmap baselines is different from Segmap"
		indx = range(len(depth_baseline))

	print(depth_baseline[0])
	print(segmap_baseline[0])

	assert indx is not None, "No baselines sample found"
	for path in img_list:
		# if True:
		# try:
			idx_to_add   = np.random.choice(indx) 
			depth_to_add = depth_baseline[idx_to_add]
			seg_to_add   = segmap_baseline[idx_to_add]
			area  = area_baseline[idx_to_add]
			label = label_baseline[idx_to_add]

			add(path, seg_to_add, depth_to_add, area, label)
		# except:
		# 	continue

def check_data():		
	with h5.File(DATASET, 'a') as ds:
		depth_set = ds['depth']
		image_set = ds['image']
		seg_set = ds['seg']

		img_list = list(image_set.keys())
		for img in img_list:
			print(depth_set[img])
			print(image_set[img])
			print(seg_set[img])

def inspect_mat(link):
	print(spio.whosmat(link))
	mat = spio.loadmat(link)
	for i in range(2):
		print('*'*50)
		char = mat['charBB'][0][i]
		word = mat['wordBB'][0][i]
		imns = mat['imnames'][0][i]
		text = mat['txt'][0][i]

		print(f'\nCHAR -- type: {type(char)} -- Shape: {char.shape} -- dtype: {char.dtype}\n{char}')
		print(f'\nWORD -- type: {type(word)} -- Shape: {word.shape} -- dtype: {word.dtype}\n{word}')
		print(f'\nIMNS -- type: {type(imns)} -- Shape: {imns.shape} -- dtype: {imns.dtype}\n{imns}')
		print(f'\nTEXT -- type: {type(text)} -- Shape: {text.shape} -- dtype: {text.dtype}\n{text}')

def generate_text():
	return np.asarray(['text']*np.random.randint(low = 1, high = 7),dtype = np.dtype('<U15'))

def generate_filename():
	return np.asarray([f'{np.random.randint(low = 1, high = 25)}/baloon_gogo_1024_1{np.random.randint(low = 1, high = 999)}'],
		dtype = np.dtype('<U20'))

def generate_charBB():
	num_charBB = np.random.randint(1,25)
	return np.asarray(100*np.random.randn(2,4,num_charBB) + 512, dtype =np.float64)

def generate_wordBB():
	num_wordBB = np.random.randint(1,11)
	return np.asarray(100*np.random.randn(2,4,num_wordBB) + 512, dtype =np.float32)

def fake_mat(n=85000):
	keys = ['charBB','wordBB','imnames','txt']
	mat = {}
	for k in keys:
		mat[k] = np.zeros(shape = (1,n), dtype = np.object)

	for i in range(n):
		#dtype = float64
		mat['charBB'][0,i] = generate_charBB()
		#dtype = float32
		mat['wordBB'][0,i] = generate_wordBB()
		#dtype = <U20
		mat['imnames'][0,i] = generate_filename()
		#dtype = <U15
		mat['txt'][0,i] = generate_text()

	spio.savemat('fake.mat', mat)

def get_file_folder(folder_name):
	list_file = []
	for root,_, files in os.walk(folder_name):
		for file in files:
			if len(file) > 0:
				list_file.append(os.path.join(root, file))
	return list_file

def get_char_box(image_shape,loc,char,font):
    w_char, h_char = font.getsize(char)

    mask = np.ones(shape = (2*h_char, 2*w_char, 3), dtype = np.int8)*255
	
    mask = Image.fromarray(np.uint8(mask)).convert('RGB')
    draw = ImageDraw.Draw(mask)

    draw.text(xy = (0,0), text = char , fill = (0,0,0), font = font, align = 'center')


    np_mask = np.array(mask)
    np_mask = cv2.cvtColor(np_mask, cv2.COLOR_RGB2GRAY)

    idx = np.where(np_mask < 255)

    xmin = idx[1].min()+loc[0]
    xmax = idx[1].max()+loc[0]
    ymin = idx[0].min()+loc[1]
    ymax = idx[0].max()+loc[1]
    
    return (xmin, ymin, xmax, ymax)

def resize_img(img, width, height):
	img = img.resize((width, height), Image.ANTIALIAS)
	return img

def random_color(max_range=  50):
	return tuple(np.random.randint(low = 0, high = max_range, size = 3))

def draw_text_into_image_inline(lines, font, background, size, top_margin = 0.08, side_margin =0.08, num_text_thresh =100):
	MAX_LEN = 10

	img_draw = Image.open(background)

	# img_draw = resize_img(img_draw, W,H)
	img_w, img_h = img_draw.size
	 
	size = W//(MAX_LEN)
	size = 35
	font_type = ImageFont.truetype(font, size=size)
	# w_char, h_char = font_type.getsize(char)
	bndboxes = []
	wordBB = []
	charBB = []
	texts = []
	# x_start , x_end = int(img_w*side_margin) , int(img_w*(1-side_margin))
	# y_start , y_end = int(img_h*top_margin) , int(img_h*(1-top_margin))
	x_start , x_end = int(img_w*side_margin) , int(img_w*(1-side_margin))
	y_start , y_end = int(img_h*top_margin) , int(img_h*(1-top_margin))

	space = size//4

	# random color:
	red = random.randint(0, 25)

	draw = ImageDraw.Draw(img_draw)
	
	x_current = x_start
	y_current  = y_start

	end_of_line = False
	num_word = 0
	line = gen_1_line(lines)
	# line = gen_plate()
	# while not end_of_line:
	for i in range(0, 100):

		color = random_color()
		for word in line:

			num_word +=1
			word = unicodedata.normalize('NFC', word)

			w_word, h_word = font_type.getsize(word)
			length = len(word)
			if w_word + x_current < x_end:
				#wordBB.append([x_current, y_current, x_current + w_word, y_current + h_word])
				for char in word:
					w_char, h_char = font_type.getsize(char)
					# w_char, h_char = get_char_size(char,font_type)
					char_box = get_char_box((img_h,img_w), (x_current,y_current),char,font_type)
					draw.text(xy=(x_current, y_current), text=char, fill=color, font=font_type, align="center")
					charBB.append(char_box)
					texts.append(word)

					x_current = x_current + w_char
				if(length >0):
					char_arr = np.array(charBB[-length:])
					wordBB.append(tuple(char_arr.min(axis=0)[:2])+tuple(char_arr.max(axis=0)[2:]))

			elif w_word + x_start > x_end:
				continue
				
			x_current +=space
			

			if x_current >= x_end:
				padding_x = random.randint(0, int(0.08*img_w))
				padding_y = random.randint(0, int(0.025*img_h))
				min_x = min(np.asarray(wordBB)[:, 0])# - padding_x
				min_y = min(np.asarray(wordBB)[:, 1]) - 5
				max_x = max(np.asarray(wordBB)[:, 2])# + random.randint(0, int(0.08*img_w))
				max_y = max(np.asarray(wordBB)[:, 3]) + 5# + random.randint(0, int(0.025*img_h))
				img_draw = img_draw.crop((min_x, min_y, max_x, max_y))

				charBB = np.asarray(charBB)
				for bb in charBB:
					bb[0] = bb[0]  - min_x
					bb[1] = bb[1]  - min_y	
					bb[2] = bb[2]  - min_x	
					bb[3] = bb[3]  - min_y
				return img_draw, wordBB, charBB, texts, font_type
	   
	return img_draw, wordBB, charBB, texts, font_type


def read_file_input(path_file):
	list_lines = []
	with open(path_file, 'r', encoding="utf-8") as file:
		lines = file.readlines()
		for line in lines:
			if len(line) > 0:
				list_lines.append(line.split('\t')[-1].replace('\n', ''))
	return list_lines


def create_fake_data(img_id, i_th, list_lines, method, list_bkgr, list_font, list_size, resdir , viz , text_threshold = 200):

	size_rand = random.sample(list_size, 1)[0]
	bkgr_rand = random.sample(list_bkgr, 1)[0]
	font_rand = random.sample(list_font, 1)[0]		
	img, wordBB, charBB, texts,font_type = draw_text_into_image_inline(list_lines, font_rand, bkgr_rand, size_rand)
	# create name file 
	name_image = f'{i_th}.jpg'

	if len(texts) < 7: return name_image, wordBB, charBB, texts
	
		#blur image
	img_cv = np.array(img)
	img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

	blur_val = np.random.choice([1,3,5])
	img_cv = cv2.blur(img_cv, (blur_val, blur_val))

	#save image to mag file 
	if img_cv is not None:

		path_to_img = os.path.join(resdir, name_image)

		filename = f'{img_id}/{name_image}'
			# img.save(os.path.join(path_out, 'imgs', name_image))
		cv2.imwrite(path_to_img, img_cv)

		###############
		data = cvt_charBB2xml_data(charBB, texts)
		parse_img2xml(img_cv, data, resdir, name_image)
		# print(''.join(texts))
		with open(path_to_img.replace('.jpg', '.txt'), 'w') as f:
			f.write(''.join(texts))
		###############

		if viz:
			img_char = img_cv.copy()
			img_word = img_cv.copy()
			for box in charBB:
				cv2.rectangle(img_char, (box[0], box[1]) , (box[2], box[3]), (255,0,0), 1)
			for box in wordBB:
				cv2.rectangle(img_word, (box[0], box[1]) , (box[2], box[3]), (255,0,0), 1)

			cv2.imwrite(os.path.join('.', 'debug', 'charBB', 'char_'+ name_image), img_char)
			cv2.imwrite(os.path.join('.', 'debug', 'wordBB', 'word_'+ name_image), img_word)

		
		# return filename, word_BB, char_BB, texts
		return filename, wordBB, charBB, texts

	else:
		raise Exception('Image is None')

def rand_number():
	ls_number = set()
	while len(ls_number) < 500000:
		num = random.randint(1, 999999)
		ls_number.add(str(num))
	return ls_number

def gen_text(corpus = None):
	assert corpus is not None, 'A Corpus is required'
	return read_file_input(corpus)

def gen_1_line(lines):
	x = np.random.randint(0, len(lines) - 1)
	return lines[x].split(' ')

def data2mat(filename ,image_names, charBBs, wordBBs, texts ):
	#[[[x_min, x_max, x_max, x_min],
	#  [y_min, y_min, y_max, y_max]], ... ]
	#
	assert '.mat' in filename, 'file must be a *.mat file'
	keys = ['charBB','wordBB','imnames','txt']

	num_samples = len(image_names)
	assert len(image_names) == len(charBBs) == len(wordBBs) == len(texts),'There must be same number of image_names, char boxes, word boxes and texts'
	
	mat = {}
	for k in keys:
		mat[k] = np.zeros(shape = (1,num_samples), dtype = np.object)

	for i in range(num_samples):
		mat['charBB'][0,i] = np.asarray(charBBs[i], dtype =np.float64)
		mat['wordBB'][0,i] = np.asarray(wordBBs[i], dtype =np.float64)
		mat['imnames'][0,i] =np.asarray(image_names[i], dtype = np.str)
		mat['txt'][0,i] = np.asarray(texts[i], dtype = np.dtype('<U15'))
		
	spio.savemat(filename, mat)


def convert_point(sample_boxes):
	#Covert points from shape list (n_sample,n_boxes, 4) to shape (n_sample,2,4,n_boxes)
	new_sample_boxes = []
	for sample in sample_boxes:
		new_box_list = [] 
		#Each sample represent and image
		for xmin, ymin, xmax, ymax in sample:
			new_box = np.zeros((4, 2), dtype='int')
			new_box[0] = [xmin, ymin]
			new_box[1] = [xmax, ymin]
			new_box[2] = [xmax, ymax]
			new_box[3] = [xmin, ymax]
			new_box_list.append(new_box)

		new_box_list = np.stack(new_box_list, axis = 0)
		new_box_list = np.transpose(new_box_list ,(2,1,0))
		new_sample_boxes.append(np.array(new_box_list))
	return new_sample_boxes


def checkbox_SynthText(link, numcheck = 10):
	import numpy as np

	gt_link = link +'/gt.mat'

	def point_inversion(points):
		points = np.transpose(points, (2,1,0))
		return points

	print(spio.whosmat(gt_link))
	mat = spio.loadmat(gt_link)


	idx_list = [i for i in range(len(mat['imnames'][0]))]
	print(idx_list)
	chosen_idx = np.random.choice(idx_list, numcheck)

	for i in chosen_idx:
		print('*'*50)
		charBB = mat['charBB'][0][i]
		imns = mat['imnames'][0][i][0]

		image = cv2.imread(link + f'/{imns}')
		print(imns)
		print(charBB.shape)

		charBB = point_inversion(charBB)
		for box in charBB:
			start_point = int(box[0][0]), int(box[0][1])
			end_point = int(box[2][0]), int(box[2][1])

			image = cv2.rectangle(image,  start_point, end_point,  (0,0,255), 1)

		resdir = f"result_{imns.split('/')[-1]}"
		cv2.imwrite(resdir, image)


def checktext_SynthText(link, numcheck = 2):
	import numpy as np

	gt_link = link +'/gt.mat'

	def point_inversion(points):
		points = np.transpose(points, (2,1,0))
		return points

	print(spio.whosmat(gt_link))
	mat = spio.loadmat(gt_link)


	idx_list = [i for i in range(len(mat['imnames'][0]))]

	for i in idx_list:
		print('*'*50)
		# charBB = mat['charBB'][0][i]
		imns = mat['imnames'][0][i][0]
		text = mat['txt'][0][i]

		color = np.random.choice([Fore.RED, Fore.BLUE, Fore.GREEN])
		print(f'{color}{imns}{Style.RESET_ALL}')
		print(f'{color}{text}{Style.RESET_ALL}')

def write_xml(data, path_xml):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = str(data["folder"])
    ET.SubElement(root, "filename").text = str(data["filename"])
    ET.SubElement(root, "path").text = str(data["path"])
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(root, "size")
    ET.SubElement(root, "segmented").text = "0"

    ET.SubElement(size, "width").text = str(data["size"]["width"])
    ET.SubElement(size, "height").text = str(data["size"]["height"])
    ET.SubElement(size, "depth").text = str(data["size"]["depth"])

    for bnbox in data["bndbox"]:
        object_doc = ET.SubElement(root, "object")
        ET.SubElement(object_doc, "name").text = str(bnbox["name"])
        ET.SubElement(object_doc, "pose").text = "Unspecified"
        ET.SubElement(object_doc, "truncated").text = "0"
        ET.SubElement(object_doc, "difficult").text = "0"
        bndboxs = ET.SubElement(object_doc, "bndbox")
        ET.SubElement(bndboxs, "xmin").text = str(bnbox["xmin"])
        ET.SubElement(bndboxs, "ymin").text = str(bnbox["ymin"])
        ET.SubElement(bndboxs, "xmax").text = str(bnbox["xmax"])
        ET.SubElement(bndboxs, "ymax").text = str(bnbox["ymax"])
    tree = ET.ElementTree(root)
    tree.write(path_xml)

def parse_img2xml(img, cands, xml_dir, name='', labels=[]):
    data = {
        "folder": 'Dataset',
        "filename": name,
        "path": os.path.join('img_dir', name),
        "size": {
            "width": img.shape[1],
            "height": img.shape[0],
            "depth": 1
        },
        "bndbox": []
    }
    
    for i, (x, y, x_max, y_max, text) in enumerate(cands):
        data["bndbox"].append({
            "name": text,
            "xmin": x,
            "ymin": y,
            "xmax": x_max,
            "ymax": y_max
        })
       
    write_xml(data, os.path.join(xml_dir, name.replace(".jpg", ".xml")))

def cvt_charBB2xml_data(charBB, texts):
	data = []
	for i, bb in enumerate(charBB):
		xmin, ymin, xmax, ymax = bb
		text = 'digit' if texts[i].isdigit() else 'char'
		data.append([xmin, ymin, xmax, ymax, text])
	return data




@timing
def GenSynthText(corpus_path,fonts,num_samples, max_samples , viz):
	# lines = gen_text('data/cleaned_VNESEcorpus_v2.txt')
	lines = gen_text(corpus_path)

	corpus_name = corpus_path.split('/')[-1].replace('.txt','')
	list_sizes = [50, 40,30,20]
	method = 'standard'

	dataSet_name = f'Synth{num_samples//1000}k_{corpus_name}' if num_samples//1000 > 0 else f'Synth{num_samples}_{corpus_name}'
	print(f'{Fore.RED}Generating {dataSet_name}{Style.RESET_ALL}')

	folder_bkgr = './data/paper_texture'
	folder_out = f'./{dataSet_name}'

	if not os.path.exists(folder_out):
		os.mkdir(folder_out)

	ls_bkgr = get_file_folder(folder_bkgr)
	ls_font = get_file_folder(fonts)

	image_names =[] 
	charBBs = []
	wordBBs= []
	texts = []

	i_th = 0
	pbar = tqdm()
	pbar.reset(total = num_samples)
	while i_th < num_samples:
		try:
		# if True:
			# print(f'{Fore.GREEN}Generating image {i_th} th{Style.RESET_ALL}')
			idx = i_th//max_samples
			folder_name = os.path.join(folder_out, str(idx))
			if not os.path.exists(folder_name):
				os.mkdir(folder_name)
			filename, wordBB, charBB, text = create_fake_data(idx,i_th, lines, method, ls_bkgr, ls_font, list_sizes, folder_name, viz = viz)

			image_names.append(filename)
			charBBs.append(charBB)
			wordBBs.append(wordBB)
			texts.append(text)
			i_th +=1
			pbar.update()
		except:
			continue
	pbar.refresh()
	# charBBs = convert_point(charBBs)
	# wordBBs = convert_point(wordBBs)

	# data2mat(os.path.join(folder_out,'gt.mat') ,image_names, charBBs, wordBBs, texts)
	return folder_out

def main(argv):
	assert FLAGS.corpus != '', f'{Fore.RED}Corpus is required, set corpus path using --corpus{Style.RESET_ALL}'
	assert FLAGS.corpus != '', f'{Fore.RED}Fonts are required, set fonts path using --fonts{Style.RESET_ALL}'

	if os.path.isfile(FLAGS.corpus):
		Synthfolder = GenSynthText(
			FLAGS.corpus,
			FLAGS.fonts,
			FLAGS.num_samples,
			FLAGS.max_samples, 
			FLAGS.viz)
	else:
		raise FileNotFoundError(f'{Fore.RED}Can not found corpus file at {FLAGS.corpus}{Style.RESET_ALL}')

if __name__ == '__main__':
	app.run(main)