def soft_split_dataset(dataset, sequences, max_timestamp_diff, max_timestamp_frame_diff, max_length):
	'''
	Split the dataset in a list of sequences, where we assume that each one of them
	can either contain only refill images or only non-refill images.
	The 'soft' split is different from the 'hard' split which requires
	'neutral' images to recognize more significative sequences
	NOTE: given dataset has 'refill' and 'None' values
	'''
	sequence = []
	timestamp = 0
	prev_timestamp = 0
	length = 0
	for item in dataset:
		if timestamp == 0:
			sequence.append(item)
			timestamp = item.timestamp
			prev_timestamp = item.timestamp
			length = 1
		elif (item.timestamp - timestamp <= max_timestamp_diff or item.timestamp - prev_timestamp <= max_timestamp_frame_diff) and length < max_length:
			sequence.append(item)
			prev_timestamp = item.timestamp
			length += 1
		else:
			sequences.append((sequence, None))
			sequence = []
			sequence.append(item)
			timestamp = item.timestamp
			prev_timestamp = item.timestamp
			length = 1

def hard_split_dataset(dataset, sequences, max_timestamp_diff, max_timestamp_frame_diff, max_length):
	'''
	Split the dataset in a list of sequences, using neutral and refill images as a splitting criterion
	The 'hard' split is different from the 'soft' split which recognizes only refill and non-refill sequences
	NOTE: given dataset has 'refill', 'neutral', 'positive' and 'negative' values
	'''
	# TODO call soft_split_dataset while not implemented
	soft_split_dataset(dataset, sequences, max_timestamp_diff, max_timestamp_frame_diff, max_length)

def predict_sequence(sequence):
	'''
	Use predicted classes of the ordered images in a sequence to associate the sequence to a class
	NOTE: given sequence is a sublist of the dataset with predicted classes all different from 'None',
	returned predicted class is a string which can be 'refill', 'neutral', 'positive' or 'negative'
	'''
	first = None
	last = None
	for item in sequence:
		if item.predicted_class == 'refill':
			return 'refill'
		if item.predicted_class == 'positive':
			if first == None:
				first = 'positive'
			last = 'positive'
		elif item.predicted_class == 'negative':
			if first == None:
				first = 'negative'
			last = 'negative'
	if first == last:
		return 'neutral'
	return last