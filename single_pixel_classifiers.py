import numpy as np # linear algebra
import os
import matplotlib.pyplot as plt
import tifffile as tiff
import sys
from datetime import datetime

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

# def sklearn_classifiers():

import create_submission


if 1:
	labels_folder = 'align_cache/labels'
	images_folder = 'align_cache/images'
else:
	labels_folder = 'cache/labels'
	images_folder = 'cache/images'

def flat_images_score(labels, predictions):
	intersection = np.sum(labels * predictions)
	# print(intersection)
	count_labels = np.sum(labels)
	count_predictions = np.sum(predictions)
	# print(count_labels, count_predictions)

	result = intersection / (count_predictions + count_labels - intersection)

	return result

def get_flat_train_data_labels(image_ids):
	label_images = np.array(list(map(lambda fn: tiff.imread(labels_folder + '/' + fn + '.tif'), image_ids)))
	input_images = np.array(list(map(lambda fn: tiff.imread(images_folder + '/' + fn + '.tif'), image_ids)))

	flat_labels = np.reshape(np.transpose(label_images, (0, 2, 3, 1)), (-1, label_images.shape[1]))
	flat_inputs = np.reshape(np.transpose(input_images, (0, 2, 3, 1)), (-1, input_images.shape[1]))

	return flat_inputs, flat_labels

def get_test_inputs(image_ids):
	input_images = np.array(list(map(lambda fn: tiff.imread(images_folder + '/' + fn + '.tif'), image_ids)))
	return input_images

def test_single_pixels():

	train_labels = os.listdir(labels_folder)

	start = datetime.now()
	label_images = np.array(list(map(lambda fn: tiff.imread(labels_folder + '/' + fn), train_labels)))
	input_images = np.array(list(map(lambda fn: tiff.imread(images_folder + '/' + fn), train_labels)))

	flat_labels = np.reshape(np.transpose(label_images, (0, 2, 3, 1)), (-1, label_images.shape[1]))
	flat_inputs = np.reshape(np.transpose(input_images, (0, 2, 3, 1)), (-1, input_images.shape[1]))

	guesses = np.random.randint(0, 2, flat_labels.shape)
	print('random guesses score: ', flat_images_score(flat_labels, guesses))

	all_predictions = np.zeros_like(flat_labels)
	all_labels = np.zeros_like(flat_labels)
	processed_count = 0


	skf = KFold(n_splits=3, shuffle=True)
	for i, (train_index, test_index) in enumerate(skf.split(flat_inputs, flat_labels)):
		print('running fold %s / %s ' % (i+1, skf.get_n_splits()))

		train_inputs, test_inputs = flat_inputs[train_index], flat_inputs[test_index]
		train_outputs, test_outputs = flat_labels[train_index], flat_labels[test_index]
		if 0:
			cls = RandomForestClassifier(verbose=True)
			cls.fit(train_inputs, train_outputs)
			predictions = cls.predict(test_inputs)
		
		else:
			predictions = np.zeros_like(test_outputs)
			for j in range(train_outputs.shape[-1]):
				cls = GaussianNB()		
				cls.fit(train_inputs, train_outputs[:,j])
				predictions[:,j] = cls.predict(test_inputs)



		all_predictions[processed_count:processed_count+len(predictions)] = predictions
		all_labels[processed_count:processed_count+len(predictions)] = test_outputs
		processed_count += len(predictions)

	print('score: ', flat_images_score(all_labels, all_predictions))

def test_image_id_split():

	labels = list(map(lambda s: s.split('.')[0], os.listdir(labels_folder)))

	all_predictions = None
	all_labels = None

	skf = KFold(n_splits=2, shuffle=True)
	for i, (train_index, test_index) in enumerate(skf.split(labels)):
		print('running fold %s / %s ' % (i+1, skf.get_n_splits()))

		train_labels = [labels[i] for i in train_index]
		test_labels = [labels[i] for i in test_index]

		train_inputs, train_outputs = get_flat_train_data_labels(train_labels)
		test_inputs, test_outputs = get_flat_train_data_labels(test_labels)

		if 0:
			predictions = np.zeros_like(test_outputs)
			for j in range(train_outputs.shape[-1]):
				cls = GaussianNB()		
				cls.fit(train_inputs, train_outputs[:,j])
				predictions[:,j] = cls.predict(test_inputs)
		else:
			cls = RandomForestClassifier(verbose=True, n_estimators=10)
			cls.fit(train_inputs, train_outputs)
			predictions = cls.predict(test_inputs)


		print('fold score: ', flat_images_score(test_outputs, predictions))

		if all_predictions is None:
			all_predictions = predictions
			all_labels = test_outputs
		else:
			all_predictions = np.concatenate((all_predictions, predictions))
			all_labels = np.concatenate((all_labels, test_outputs))

	print('score: ', flat_images_score(all_labels, all_predictions))

def make_single_pixel_submission():
	train_image_ids = list(map(lambda s: s.split('.')[0], os.listdir(labels_folder)))
	test_image_ids = create_submission.get_test_image_ids()

	train_inputs, train_outputs = get_flat_train_data_labels(train_image_ids)

	print('training classifier...')
	cls = RandomForestClassifier(verbose=False, n_estimators=10)	
	cls.fit(train_inputs, train_outputs)

	test_inputs = get_test_inputs(test_image_ids)

	test_predictions = np.zeros((test_inputs.shape[0], 10, test_inputs.shape[2], test_inputs.shape[3]))
	print('making test predictions...')

	for i, (image_id, test_image) in enumerate(zip(test_image_ids, test_inputs)):
		sys.stdout.write('\b\b\b\b\b\b\b\b\b\b\b\b%d/%d' % (i, len(test_inputs))); sys.stdout.flush()
		test_image_flat = np.reshape(np.transpose(test_image, (1, 2, 0)), (-1, test_image.shape[0]))

		predictions_flat = cls.predict(test_image_flat)

		predictions = np.transpose(np.reshape(predictions_flat, (test_image.shape[1], test_image.shape[2], predictions_flat.shape[1])), (2, 0, 1))
		
		if 0:
			for i in range(10):
				plt.subplot(2, 5, i+1); plt.imshow(predictions[i])

			plt.title(image_id)
			plt.show()

		test_predictions[i] = predictions

	print('\ncreating submission file...')
	create_submission.create_submission_file(test_image_ids, test_predictions, 'random forest single pixel_aligned.csv', simplify=False)

def test_unflatten():
	image_ids = ['6010_1_2', '6010_4_2']

	label_images = np.array(list(map(lambda fn: tiff.imread(labels_folder + '/' + fn + '.tif'), image_ids)))
	input_images = np.array(list(map(lambda fn: tiff.imread(images_folder + '/' + fn + '.tif'), image_ids)))

	flat_labels = np.reshape(np.transpose(label_images, (0, 2, 3, 1)), (-1, label_images.shape[1]))
	flat_inputs = np.reshape(np.transpose(input_images, (0, 2, 3, 1)), (-1, input_images.shape[1]))

	test_image = label_images[0]
	print(test_image.shape)

	test_image_flat = np.reshape(np.transpose(test_image, (1, 2, 0)), (-1, test_image.shape[0]))	

	predictions_flat = test_image_flat

	print(predictions_flat.shape)

	predictions = np.transpose(np.reshape(predictions_flat, (test_image.shape[1], test_image.shape[2], predictions_flat.shape[1])), (2, 0, 1))

	print(predictions.shape)

	for i in range(10):
		plt.subplot(1, 2, 1); plt.imshow(test_image[i])
		plt.subplot(1, 2, 2); plt.imshow(predictions[i])
		plt.show()

def test_flatten():

	image_ids = list(map(lambda s: s.split('.')[0], os.listdir(labels_folder)))

	label_images = np.array(list(map(lambda fn: tiff.imread(labels_folder + '/' + fn + '.tif'), image_ids)))
	input_images = np.array(list(map(lambda fn: tiff.imread(images_folder + '/' + fn + '.tif'), image_ids)))

	flat_labels = np.reshape(np.transpose(label_images, (0, 2, 3, 1)), (-1, label_images.shape[1]))
	flat_inputs = np.reshape(np.transpose(input_images, (0, 2, 3, 1)), (-1, input_images.shape[1]))

	print(flat_labels[:20])
	print(flat_labels.shape, label_images.shape)
	print(label_images[0, :, 0, :20])

if __name__ == "__main__":
	test_flatten()
	# test_single_pixels()
	# test_image_id_split()
	# make_single_pixel_submission()
	# test_unflatten()
