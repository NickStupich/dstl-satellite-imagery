import numpy as np # linear algebra
import os
import matplotlib.pyplot as plt
import tifffile as tiff
import sys
from datetime import datetime
from sklearn.model_selection import KFold

import create_submission

np.random.seed(1)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D

labels_folder = 'cache/labels'
images_folder = 'cache/images'

def get_train_data_labels(image_ids, padding = 0):
	label_images = np.array(list(map(lambda fn: tiff.imread(labels_folder + '/' + fn + '.tif'), image_ids)))
	input_images = np.array(list(map(lambda fn: tiff.imread(images_folder + '/' + fn + '.tif'), image_ids)))

	#TODO: subtract channel means
	if padding > 0:
		padded_input_images = np.zeros((input_images.shape[0], input_images.shape[1], input_images.shape[2]+2*padding, input_images.shape[3]+2*padding))
		padded_input_images[:, :, padding:-padding, padding:-padding] = input_images
	else:
		padded_input_images = input_images

	print(np.max(padded_input_images))
	padded_input_images = padded_input_images / (255.0 * 64) - 0.1
	print(np.max(padded_input_images), np.mean(padded_input_images))

	return padded_input_images, label_images

def single_layer_score(labels, predictions):
	flat_labels = labels.flatten()
	flat_predictions = predictions.flatten()

	intersection = np.sum(flat_labels * flat_predictions)
	count_labels = np.sum(flat_labels)
	count_predictions = np.sum(flat_predictions)

	result = intersection / (count_predictions + count_labels - intersection)

	return result


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
	padding = 0
	model = Sequential()
	# model.add(Convolution2D(32, 5, 5, border_mode='same', input_shape=input_shape, activation='sigmoid'))
	# model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode='same'))
	# model.add(Convolution2D(1, 5, 5, border_mode='same', activation='sigmoid'))

	# model.add(Convolution2D(100, 1, 1, border_mode='same', input_shape=input_shape, activation='sigmoid'))
	# model.add(Convolution2D(100, 1, 1, border_mode='same', activation='sigmoid'))

	model.add(Convolution2D(1, 1, 1, border_mode='same', activation='linear',  input_shape=input_shape))


	model.add(Reshape((input_shape[1] - 2*padding, input_shape[2] - 2*padding)))


	# model.add(Activation('softmax'))

	# model.compile(loss='binary_crossentropy', optimizer='sgd')#, metrics=['mse, accuracy'])
	model.compile(loss='mse', optimizer='rmsprop')#, metrics=['mse, accuracy'])

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

		padding=0
		train_inputs, train_outputs = get_train_data_labels(train_labels, padding=padding)
		test_inputs, test_outputs = get_train_data_labels(test_labels, padding=padding)

		# train_inputs = np.tranpose(train_inputs.tranpose((0, 2, 3, 1))
		# test_inputs = np.tranpose(test_inputs, (0, 2, 3, 1))

		print(train_inputs.shape)

		if 0:
			model = get_model_all_outputs(train_inputs.shape[1:])
			model.fit(train_inputs, train_outputs, batch_size=1, nb_epoch=50, verbose=1)

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
				label_outputs = train_outputs[:, label, :, :]
				print('label outputs: ', label_outputs.shape)
				model = get_model_single_outputs(train_inputs.shape[1:])
				model.fit(train_inputs, label_outputs, nb_epoch=10)

				predictions = model.predict(test_inputs)
				score = single_layer_score(test_outputs[:, label], predictions)
				print('label %d score: ' % label, score)

				train_predictions = model.predict(train_inputs)
				print(train_predictions.shape, label_outputs.shape)

				sample_index = 0
				print(test_outputs.shape, predictions.shape)

				# plt.subplot(1, 2, 1)
				# plt.imshow(label_outputs[sample_index])
				# plt.subplot(1, 2, 2)
				# plt.imshow(train_predictions[sample_index])
				# plt.show()

				for i in range(10):
					plt.subplot(3, 4, i+1)
					plt.imshow(train_outputs[sample_index, i, :, :])

				plt.subplot(3, 4, 12)
				plt.title('predictions')
				plt.imshow(train_predictions[sample_index])

				plt.show()


				# if all_predictions is None:
				# 	all_predictions = predictions
				# 	all_labels = test_outputs
				# else:
				# 	all_predictions = np.concatenate((all_predictions, predictions))
				# 	all_labels = np.concatenate((all_labels, test_outputs))

	print('overall score: ', images_score(all_labels, all_predictions))


if __name__ == "__main__":
	test_image_id_split()