import numpy as np # linear algebra
import os
import matplotlib.pyplot as plt
import tifffile as tiff
import sys
from datetime import datetime
from sklearn.model_selection import KFold

import create_submission

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D

labels_folder = 'cache/labels'
images_folder = 'cache/images'

def get_train_data_labels(image_ids, padding = 0):
	label_images = np.array(list(map(lambda fn: tiff.imread(labels_folder + '/' + fn + '.tif'), image_ids)))
	input_images = np.array(list(map(lambda fn: tiff.imread(images_folder + '/' + fn + '.tif'), image_ids)))

	#TODO: subtract channel means

	padded_input_images = np.zeros((input_images.shape[0], input_images.shape[1], input_images.shape[2]+2*padding, input_images.shape[3]+2*padding))
	padded_input_images[:, :, padding:-padding, padding:-padding] = input_images

	return padded_input_images, label_images

def single_layer_score(labels, predictions):
	flat_labels = labels.flatten()
	flat_predictions = predictions.flatten()

	intersection = np.sum(labels_flat * predictions_flat)
	count_labels = np.sum(labels_flat)
	count_predictions = np.sum(predictions_flat)
	

def images_score(labels, predictions):

	labels_flat = np.reshape(np.transpose(labels, (0, 2, 3, 1)), (-1, labels.shape[1]))
	predictions_flat = np.reshape(np.transpose(predictions, (0, 2, 3, 1)), (-1, predictions.shape[1]))

	intersection = np.sum(labels_flat * predictions_flat)
	count_labels = np.sum(labels_flat)
	count_predictions = np.sum(predictions_flat)

	result = intersection / (count_predictions + count_labels - intersection)

	return result

def get_model_all_outputs(input_shape):
	model = Sequential()
	model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape, activation='sigmoid'))
	model.add(Convolution2D(10, 3, 3, border_mode='valid', input_shape=input_shape, activation='sigmoid'))
	

	# model.add(Activation('softmax'))

	model.compile(loss='binary_crossentropy', optimizer='sgd')#, metrics=['mse, accuracy'])

	model.summary()

	return model

def get_model_single_outputs(input_shape):
	padding = 2
	model = Sequential()
	model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape, activation='sigmoid'))
	model.add(Convolution2D(1, 3, 3, border_mode='valid', input_shape=input_shape, activation='sigmoid'))

	model.add(Reshape((input_shape[1] - 2*padding, input_shape[2] - 2*padding)))

	# model.add(Activation('softmax'))

	model.compile(loss='binary_crossentropy', optimizer='sgd')#, metrics=['mse, accuracy'])

	model.summary()

	return model

def test_image_id_split():

	labels = list(map(lambda s: s.split('.')[0], os.listdir(labels_folder)))

	all_predictions = None
	all_labels = None

	skf = KFold(n_splits=2, shuffle=True)
	for i, (train_index, test_index) in enumerate(skf.split(labels)):
		print('running fold %s / %s ' % (i+1, skf.get_n_splits()))

		train_labels = [labels[i] for i in train_index]
		test_labels = [labels[i] for i in test_index]

		padding=2
		train_inputs, train_outputs = get_train_data_labels(train_labels, padding=padding)
		test_inputs, test_outputs = get_train_data_labels(test_labels, padding=padding)

		if 0:
			model = get_model_all_outputs(train_inputs.shape[1:])
			model.fit(train_inputs, train_outputs, batch_size=1, nb_epoch=100, verbose=1)

			predictions = model.predict(test_inputs)
			score = images_score(test_outputs, predictions)
			print('fold score: ', score)

			if all_predictions is None:
				all_predictions = predictions
				all_labels = test_outputs
			else:
				all_predictions = np.concatenate((all_predictions, predictions))
				all_labels = np.concatenate((all_labels, test_outputs))

		else:
			for label in range(10):
				model = get_model_single_outputs(train_inputs.shape[1:])
				model.fit(train_inputs, train_outputs[:, label, :, :])

				predictions = model.predict(test_inputs)
				score = single_layer_score(test_outputs[:, label], predictions)
				print('label %d score: ' % label, score)

				# if all_predictions is None:
				# 	all_predictions = predictions
				# 	all_labels = test_outputs
				# else:
				# 	all_predictions = np.concatenate((all_predictions, predictions))
				# 	all_labels = np.concatenate((all_labels, test_outputs))

	print('overall score: ', images_score(all_labels, all_predictions))


if __name__ == "__main__":
	test_image_id_split()