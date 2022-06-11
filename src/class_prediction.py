from train_alexnet import getAlexNet
from train_caffenet import getcaffenet
from train_cnn import getCNN
from train_cnn2 import getcnn2

NN_TYPES = {
	'cnn' : getCNN,
	'cnn2' : getcnn2,
	'alexnet' : getAlexNet,
	'caffenet' : getcaffenet
}

def get_model(weights, nn_type = 'alexnet'):
	global NN_TYPES
	model = NN_TYPES[nn_type]()
	model.load_weights(weights)
	return model


# Use the neural network model (nn_model) to predict refill images from non-refill images,
# the predicted class of the dataset items will be set to 'refill' for those predicted images
# NOTE: dataset has 'None' values before; 'refill' and 'None' values after
def predict_refill(dataset, nn_model):
	model = get_model(nn_model)
	for item in dataset:
		item.predict('refill', None, model)

# Use the neural network model (nn_model) and the rgb images to predict neutral images from non-neutral images,
# return a list with the predicted class for each element of the dataset set to 'neutral' or 'None'
def predict_neutral_from_rgb(dataset, nn_model):
	predictions = []
	model = get_model(nn_model)
	for item in dataset:
		predictions.append(item.predict(None, 'neutral', model, voting = True))
	return predictions

# Use the neural network model (nn_model) and the depth images to predict neutral images from non-neutral images,
# return a list with the predicted class for each element of the dataset set to 'neutral' or 'None'
def predict_neutral_from_depth(dataset, nn_model):
	predictions = []
	model = get_model(nn_model)
	for item in dataset:
		predictions.append(item.predict(None, 'neutral', model, voting = True, img_type = 'depth'))
	return predictions

def predict_neutral_from_prep(dataset, nn_model):
	predictions = []
	model = get_model(nn_model)
	for item in dataset:
		predictions.append(item.predict(None, 'neutral', model, voting = True, img_type = 'prep'))
	return predictions

# Use the neural network model (nn_model) and the preprocessed rgb images to predict positive and negative images,
# the predicted class of the dataset items will be set to either 'positive' or 'negative'
# NOTE: dataset has 'None' values before; 'positive' and 'negative' values after
def predict_posneg(dataset, nn_model):
	model = get_model(nn_model)
	for item in dataset:
		item.predict('positive', 'negative', model)

def predict_posneg_mask(dataset, nn_model):
	model = get_model(nn_model)
	for item in dataset:
		item.predict('positive', 'negative', model, img_type = 'prep')