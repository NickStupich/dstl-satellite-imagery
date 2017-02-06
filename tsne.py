import numpy as np # linear algebra
import os
import matplotlib.pyplot as plt
import tifffile as tiff
import sys
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

from sklearn.manifold import TSNE

import single_pixel_classifiers

if 0:
	labels_folder = 'align_cache/labels'
	images_folder = 'align_cache/images'
else:
	labels_folder = 'cache/labels'
	images_folder = 'cache/images'

def get_class_numbers(outputs):
	result = np.argmax(outputs, axis=1)
	print(outputs[:20])
	print(result.shape)

	sums = np.sum(outputs, axis=1)
	nothings = np.where(sums == 0)[0]
	print(nothings.shape)
	result[nothings] == outputs.shape[1]

	return result

if __name__ == "__main__":
	labels = list(map(lambda s: s.split('.')[0], os.listdir(labels_folder)))


	train_inputs, train_outputs = single_pixel_classifiers.get_flat_train_data_labels(labels)

	skip = 1000
	train_inputs = train_inputs[::skip, :]
	train_outputs = train_outputs[::skip, :]

	outputs_dense = get_class_numbers(train_outputs)
	print(outputs_dense.shape)

	print(train_inputs.shape, train_outputs.shape)

	dimensions = 3
	tsne = TSNE(n_components=dimensions)

	transformed_inputs = tsne.fit_transform(train_inputs)

	print(transformed_inputs.shape)

	if dimensions == 3:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(transformed_inputs[:, 0], transformed_inputs[:, 1], transformed_inputs[:, 2], c=outputs_dense)

	else:
		plt.scatter(transformed_inputs[:,0], transformed_inputs[:, 1], c=outputs_dense)
		plt.grid(True)
	
	plt.show()