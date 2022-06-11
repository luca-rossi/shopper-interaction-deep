# This script gets as input a sequence of frames (both RGB and depth images) with their timestamps and labels
# and follows a 3-steps process to classify them as positive, negative, neutral or refill.
# For each class some evaluation metrics are computed, such as accuracy, precision, recall and f1-score
# At the end we propose an introduction to a sequential approach, so the dataset is divided into sequences,
# and those sequences are classified as positive (a product has been taken), negative (a product has been put back),
# neutral or refill. Since this is only a first approach, no evaluation metrics are computed

# NOTE1: make sure that the input test set doesn't contain data used to train the neural networks!!!
# NOTE2: the predicted class for a single image doesn't depend only on the image itself,
#	but on other images too (sequences of refill)
# NOTE3: when splitting and predicting sequences, remember that refill sequences have already been found
# NOTE4: the test set (NOT the training set) has to be relabeled to include refill images that have been excluded before
# NOTE5: the neural networks must be trained BEFORE running this script, the results will be shown and confronted separately


from item import *
from utils import *
from sequences import *
from preprocessing import *
from class_prediction import *


# Parameters
CSV_PATH = 'data.csv'
OUTPUT_PATH = 'res.csv'
NN_REFILL = 'weights/refill.h5'
NN_NEUTRAL_RGB = 'weights/neutral rgb.h5'
NN_NEUTRAL_DEPTH = 'weights/neutral depth.h5'
NN_NEUTRAL_DEPTH_THR = 'weights/neutral thr.h5'
NN_NEUTRAL_DEPTH_THR_LOW = 'weights/neutral low.h5'
NN_NEUTRAL_DEPTH_THR_MASK = 'weights/neutral mask.h5'
NN_POSNEG = 'weights/posneg rgb.h5'
NN_POSNEG_MASK = 'weights/posneg mask.h5'

MAX_TIMESTAMP_DIFF = 60000			# 2 min
MAX_TIMESTAMP_FRAMES_DIFF = 3000
MAX_LENGTH = 100
MIN_REFILL = 2
MIN_REFILL_LENGTH = 5
MIN_REFILL_RATIO = 0.2
MIN_NEUTRALS_FOR_VOTING = 2
DARK_THRESHOLD = 40
DARK_RATIO = 98

dataset = []
sequences = []

# Get input
get_input(dataset, CSV_PATH)

# STEP 1: predict refill (neural network and sequence recognition)
print('Predicting refill class...')
predict_refill(dataset, NN_REFILL)
soft_split_dataset(dataset, sequences, MAX_TIMESTAMP_DIFF, MAX_TIMESTAMP_FRAMES_DIFF, MAX_LENGTH)
for sequence, prediction in sequences:
	prediction = predict_refill_sequence(sequence, MIN_REFILL, MIN_REFILL_RATIO, MIN_REFILL_LENGTH)
reduced_dataset = reduce_dataset(dataset)

# STEP 2: predict neutral (voting system combining neural networks and feature-based approaches,
#							some preprocessing for the next step is also performed)
##........................................................................................
print('Preparing for preprocessing...')
prepare_for_preprocessing(reduced_dataset)

print('Predicting neutral class...')
print('... from rgb...')
predictions_rgb = predict_neutral_from_rgb(reduced_dataset, NN_NEUTRAL_RGB)
print('... from depth...')
predictions_depth = predict_neutral_from_depth(reduced_dataset, NN_NEUTRAL_DEPTH)
print('... from low threshold...')
preprocess_thr_low(reduced_dataset)
predictions_thr_low = predict_neutral_from_prep(reduced_dataset, NN_NEUTRAL_DEPTH_THR_LOW)
print('... from high and low threshold...')
preprocess_thr(reduced_dataset)
predictions_thr = predict_neutral_from_prep(reduced_dataset, NN_NEUTRAL_DEPTH_THR)
print('... from masked rgbs...')
preprocess_mask(reduced_dataset)
predictions_mask = predict_neutral_from_prep(reduced_dataset, NN_NEUTRAL_DEPTH_THR_MASK)
## other methods...
predictions = [predictions_rgb, predictions_depth, predictions_thr, predictions_thr_low, predictions_mask]
prediction_weights = [1,0,1,1,2]
vote(reduced_dataset, predictions, prediction_weights, MIN_NEUTRALS_FOR_VOTING)
reduced_dataset = reduce_dataset(dataset)

# STEP 3: predict positive / negative (neural network, uses the preprocessed dataset from the previous step)
print('Predicting positive and negative classes...')
preprocess_mask(reduced_dataset)				#already done
normal_dataset, masked_dataset = split_normal_and_masked(reduced_dataset, DARK_THRESHOLD, DARK_RATIO)
predict_posneg(normal_dataset, NN_POSNEG)
predict_posneg_mask(masked_dataset, NN_POSNEG_MASK)
#predict_posneg_mask(reduced_dataset, NN_POSNEG_MASK)
#predict_posneg(reduced_dataset, NN_POSNEG)

# Compute and print metrics for each class
print('Calculating metrics for each class...')
metrics(dataset, 'refill')
metrics(dataset, 'neutral')
metrics(dataset, 'positive')
metrics(dataset, 'negative')

# Predict sequences
print('Predicting sequences...')
hard_split_dataset(dataset, sequences, MAX_TIMESTAMP_DIFF, MAX_TIMESTAMP_FRAMES_DIFF, MAX_LENGTH)
for sequence, prediction in sequences:
	prediction = predict_sequence(sequence)
	print('Sequence from ' + sequence[0].filename + ' to ' + sequence[-1].filename + ': ' + prediction)

# Save results
save_results(OUTPUT_PATH, dataset)