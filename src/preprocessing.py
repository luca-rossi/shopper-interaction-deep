from keras.preprocessing import image
import numpy as np
from cv2 import *

def prepare_for_preprocessing(dataset):
	for item in dataset:
		f = item.filename.split('\\')[-1].split('/')[-1]
		img = imread('images/'+f+'.png', 0)
	
		height, width = img.shape
	
		for row in range(height):
			for col in range(width):
				try:
					img[row][col] = img[row][col] * 20
				except IndexError:
					pass

		halfmask = np.zeros((height,width), np.uint8)
		halfmask[0:int(0.5*height),:] = 0
		halfmask[int(0.5*height):height,:] = 255
		img = bitwise_and(img, img, mask = halfmask)

		mask_thr = inRange(img, 100, 200)
		img_thr = bitwise_and(img, img, mask = mask_thr)
		imwrite('images_thr/'+f+'.png', img_thr)

		mask_thr_low = inRange(img, 100, 255)
		img_thr_low = bitwise_and(img, img, mask = mask_thr_low)
		imwrite('images_thr_low/'+f+'.png', img_thr_low)
		
		mask_rgb = inRange(img, 100, 218) #237 
		img_mask_rgb = bitwise_and(img, img, mask = mask_rgb)
		ret, th = threshold(img_mask_rgb, 0, 255, THRESH_BINARY)		##???
		imgrgb = imread('images/'+f+'.jpg')
		masked = bitwise_and(imgrgb, imgrgb, mask = th)
		imwrite('images_mask/'+f+'.jpg', masked)

def preprocess_thr(dataset):
	for item in dataset:
		path = 'images_thr/' + item.filename.split('\\')[-1].split('/')[-1]
		img = image.load_img(path + '.png', target_size = (80, 80))
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis = 0)
		item.set_image_prep(img)		##TODO fix

def preprocess_thr_low(dataset):
	for item in dataset:
		path = 'images_thr_low/' + item.filename.split('\\')[-1].split('/')[-1]
		img = image.load_img(path + '.png', target_size = (80, 80))
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis = 0)
		item.set_image_prep(img)		##TODO fix

def preprocess_mask(dataset):
	for item in dataset:
		path = 'images_mask/' + item.filename.split('\\')[-1].split('/')[-1]
		img = image.load_img(path + '.jpg', target_size = (80, 80))
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis = 0)
		item.set_image_prep(img)		##TODO fix