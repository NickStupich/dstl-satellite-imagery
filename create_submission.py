

import pandas as pd
import numpy as np
import sys

import mask_to_polycon
import export_pixel_mask

from shapely.wkt import dumps

NUM_CLASSES = 10


def get_test_image_ids(sample_file_filename = 'data/sample_submission.csv'):
	f = open(sample_file_filename)
	f.readline()
	result = []
	for line in f:
		image_id = line.split(',')[0]
		if not image_id in result:
			result.append(image_id)

	# print(len(result))
	# result = list(set(result))
	# print(len(result))
	return result

def create_submission_file(image_ids, prediction_images, filename, simplify = True):
	gs = pd.read_csv('data/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

	f = open('submissions/' + filename, 'w')
	f.write('ImageId,ClassType,MultipolygonWKT\n')

	print('creating submission file...')

	for i, (image_id, predictions) in enumerate(zip(image_ids, prediction_images)):
		sys.stdout.write('\b\b\b\b\b\b\b\b\b\b\b\b\b\b%d/%d' % (i, len(image_ids))); sys.stdout.flush()

		xymax = export_pixel_mask._get_xmax_ymin(gs,image_id)
		for class_num, mask in enumerate(predictions):
			# print(image_id)
			polycons = mask_to_polycon.mask_to_polygons(mask, xymax)
			if simplify:
				polycons = polycons.simplify(0.000008, preserve_topology=False)

			# print(class_num, polycons)

			# f.write('%s,%s,"%s"\n' % (image_id, class_num+1, str(polycons)))


			f.write('%s,%s,"%s"\n' % (image_id, class_num+1, dumps(polycons, rounding_precision=10)))

	f.close()
	print('\ndone')


if __name__ == "__main__":
	test_image_ids = get_test_image_ids()

	image_size = (1000,1000)
	prediction_images = np.zeros((len(test_image_ids), NUM_CLASSES, image_size[0], image_size[1]), dtype='uint8')

	prediction_images[:,:,100:200, 200:300] = 1
	prediction_images[:,:,300:500, 700:900] = 1

	create_submission_file(test_image_ids, prediction_images, 'submission.csv')
