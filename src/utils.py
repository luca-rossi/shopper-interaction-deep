from item import *
from cv2 import *

EPS = 0.00001

def get_input(dataset, csv_path):
	'''
	Get a csv as input and creates the dataset list. Each element of this list is an istance of the Item class containing:
	- RGB image
	- Depth image
	- Timestamp
	- Predicted class (initially None)
	- Label (used to evaluate the metrics)
	'''
	with open(csv_path, 'rb') as file:
		lines = file.readlines()
		for i, line in enumerate(lines):
			if (i != 0):
				splitted_line = line.decode().split(',')
				if not 'skip' in splitted_line[1]:
					path = splitted_line[0].split('.')[0]
					filename = splitted_line[0].split('/')[-1].split('.')[0]
					label = splitted_line[1].replace('\n', '').replace('\r', '').replace('"', '')
					dataset.append(Item(filename, path, label))
	file.close()

def reduce_dataset(dataset):
	'''
	Returns a reduced dataset containing only the items of the dataset where the predicted class is None
	'''
	reduced_dataset = []
	for item in dataset:
		if item.predicted_class == None:
			reduced_dataset.append(item)
	return reduced_dataset

def vote(dataset, predictions, weights, min_neutral_for_voting):
	'''
	Get a list of lists of predicted classes for each image in the dataset,
	if at least N predictions (with N = min_posneg_for_voting) for a given image are 'posneg',
	then the final predicted class will be 'posneg', otherwise it will be 'neutral'
	'''
	n = len(predictions)
	i = 0
	for item in dataset:
		num_neutrals = 0
		for index, prediction in enumerate(predictions):
			if prediction[i] == 'neutral':
				num_neutrals += weights[index]
		if num_neutrals >= min_neutral_for_voting:
			item.predicted_class = 'neutral'
		i += 1

def predict_refill_sequence(sequence, min_refill, min_refill_ratio, min_refill_length):
	'''
	For a given sequence of images, if at least a N of them have been predicted as 'refill', with N >= min_refill_ratio,
	every other image in the same sequence will be set 'refill', otherwise every 'refill' image will be set to None
	'''
	refill = False
	num_refill = 0
	if len(sequence) >= min_refill_length:
		for item in sequence:
			if item.predicted_class == 'refill':
				num_refill += 1
		if num_refill / len(sequence) >= min_refill_ratio:# and num_refill >= min_refill:
			refill = True
	for item in sequence:
		item.predicted_class = 'refill' if refill else None
	return 'refill' if refill else None

def split_normal_and_masked(dataset, dark_threshold, dark_ratio):
	normal = []
	masked = []
	for item in dataset:
		f = item.filename.split('\\')[-1].split('/')[-1]
		img = imread('images_mask/'+f+'.jpg', 0)
		height, width = img.shape
		i = 0
		for row in range(height):
			for col in range(width):
				color = int(img[row, col])
				if color <= dark_threshold:
					i += 1
		x = (height * width) * (dark_ratio / 100)
		if i >= x:
			masked.append(item)
		else:
			normal.append(item)
	return normal, masked

def metrics(dataset, class_name):
	'''
	# For a given class name ('refill', 'neutral', 'positive' or 'negative')
	# calculate the metrics (accuracy, precision, recall, f1-score) from the dataset and print them
	# NOTE: given dataset has predicted classes all different from 'None'
	'''
	global EPS
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for item in dataset:
		if item.predicted_class == class_name:			# my class is predicted
			if class_name == item.label:					# right class picked
				tp += 1
			else:											# wrong class picked
				fp += 1
		else:											# another class is predicted
			if class_name == item.label:					# my (right) class not picked
				fn += 1
			else:											# my (wrong) class not picked
				tn += 1
	accuracy = (tp + tn) / (tp + tn + fp + fn + EPS)
	precision = tp / (tp + fp + EPS)
	recall = tp / (tp + fn + EPS)
	f1score = 2 * precision * recall / (precision + recall + EPS)
	print('************** Metrics for class ' + class_name + ' **************')
	print('Accuracy = ' + str(accuracy))
	print('Precision = ' + str(precision))
	print('Recall = ' + str(recall))
	print('F1-score = ' + str(f1score))

def save_results(filename, dataset):
	'''
	For each image in the dataset save the predicted class in an external file
	'''
	with open(filename, 'w') as file:
		for item in dataset:
			file.write(str(item.filename) + ',' + str(item.label) + ',' + str(item.predicted_class) + '\n')
	file.close()