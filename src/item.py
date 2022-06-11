from keras import backend as K
from keras.preprocessing import image
import numpy as np

IMG_SIZE = 80

class Item:

	filename = None
	image_rgb = None
	image_depth = None
	image_prep = None
	timestamp = None
	predicted_class = None
	label = None

	def __init__(self, filename, path, label):
		global IMG_SIZE
		
		self.filename = filename
		
		self.image_rgb = image.load_img(self.filename + '.jpg', target_size = (IMG_SIZE, IMG_SIZE))
		self.image_rgb = image.img_to_array(self.image_rgb)
		self.image_rgb = np.expand_dims(self.image_rgb, axis = 0)
		
		self.image_depth = image.load_img(self.filename + '.png', target_size = (IMG_SIZE, IMG_SIZE))
		self.image_depth = image.img_to_array(self.image_depth)
		self.image_depth = np.expand_dims(self.image_depth, axis = 0)
		
		self.timestamp = int(filename.split('\\')[-1].split('/')[-1][:17])		##TODO fix
		
		self.label = label

	def set_image_prep(self, image_prep):
		self.image_prep = image_prep

	def predict(self, label0, label1, model, voting = False, img_type = 'rgb'):
		img = self.image_rgb if img_type == 'rgb' else (self.image_depth if img_type == 'depth' else self.image_prep)
		result = model.predict(img)						# [0, 1] or [1, 0]
		prediction = label0 if result[0][0] == 0 else label1
		if voting:
			return prediction
		self.predicted_class = prediction