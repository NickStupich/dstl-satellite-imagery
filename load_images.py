import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import matplotlib.pyplot as plt
from subprocess import check_output
import cv2
import tifffile as tiff
import sys

import export_pixel_mask

def _align_two_rasters(img1,img2, tol=1e-3):
	p1 = img1.astype(np.float32)
	p2 = img2.astype(np.float32)

	# lp1 = cv2.Laplacian(p1,cv2.CV_32F,ksize=5)
	# lp2 = cv2.Laplacian(p2,cv2.CV_32F,ksize=5)

	warp_mode = cv2.MOTION_EUCLIDEAN
	warp_matrix = np.eye(2, 3, dtype=np.float32)
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  tol)
	(cc, warp_matrix) = cv2.findTransformECC (p1, p2,warp_matrix, warp_mode, criteria)
	print("_align_two_rasters: cc:{}".format(cc), warp_matrix)

	# img3 = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	# img3[img3 == 0] = np.average(img3)

	# return img3

def align_to_image(ref_img,image, tol=1e-5):
	p1 = ref_img.astype(np.float32)
	p2 = image.astype(np.float32)

	padding = 100

	p2_padded = np.zeros((p2.shape[0] + 2*padding, p2.shape[1]+2*padding), dtype=image.dtype)
	p2_padded[padding:p2.shape[0]+padding,padding:p2.shape[1]+padding] = p2

	match = cv2.matchTemplate(p1, p2_padded, cv2.TM_CCOEFF_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
	print(max_loc[0] - padding, max_loc[1] - padding)

	plt.imshow(match); plt.title('match template'); plt.show()

	# warp_mode = cv2.MOTION_EUCLIDEAN
	warp_mode = cv2.MOTION_TRANSLATION
	warp_matrix = np.eye(2, 3, dtype=np.float32)
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  tol)
	(cc, warp_matrix) = cv2.findTransformECC (p1, p2,warp_matrix, warp_mode, criteria)
	print("_align_two_rasters: cc:{}".format(cc), warp_matrix)

	# img3 = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	# img3[img3 == 0] = np.average(img3)

	# return img3

def get_alignment_displacement(ref_imgs, images, plot=False):
	
	padding = 100

	images_padded = np.zeros((images.shape[0], images.shape[1] + 2*padding, images.shape[2]+2*padding), dtype=images.dtype)
	images_padded[:, padding:images.shape[1]+padding,padding:images.shape[2]+padding] = images

	best_locations = []
	for i in range(ref_imgs.shape[0]):
		for j in range(images_padded.shape[0]):
			match = cv2.matchTemplate(ref_imgs[i], images_padded[j], cv2.TM_CCOEFF_NORMED)
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
			best_location = (max_loc[0] - padding, max_loc[1] - padding)
			best_locations.append(best_location)

	best_locations = np.array(best_locations)
	best_location = np.mean(best_locations, axis=0)
	# print(best_locations, best_location)

	warp_matrix = np.array([[1, 0, best_location[0]], [0, 1, best_location[1]]])

	# aligned_images = cv2.warpAffine(images[0], warp_matrix, (images.shape[2], images.shape[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	aligned_images = np.zeros_like(images)
	for i in range(images.shape[0]):
		aligned_images[i] = cv2.warpAffine(images[i], warp_matrix, (images.shape[2], images.shape[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

	if plot:
		print(best_location)
		plt.subplot(1, 2, 1); plt.imshow(images[0])
		plt.subplot(1, 2, 2); plt.imshow(aligned_images[0])
		plt.show()

	return aligned_images

def load_image_layers(image_id, size = (3348, 3403)):
	img_3 = tiff.imread("data/three_band/{}.tif".format(image_id))
	img_a_raw = tiff.imread("data/sixteen_band/{}_A.tif".format(image_id))
	img_m_raw = tiff.imread("data/sixteen_band/{}_M.tif".format(image_id))
	img_p_raw = np.array([tiff.imread("data/sixteen_band/{}_P.tif".format(image_id))])

	layers = np.zeros((20, size[0], size[1]), dtype='uint16')

	layers[0:3] = np.array([cv2.resize(img_3[i],(size[1],size[0]),interpolation=cv2.INTER_CUBIC) for i in range(img_3.shape[0])])
	layers[3:11] = np.array([cv2.resize(img_a_raw[i],(size[1],size[0]),interpolation=cv2.INTER_CUBIC) for i in range(img_a_raw.shape[0])])
	layers[11:19] = np.array([cv2.resize(img_m_raw[i],(size[1],size[0]),interpolation=cv2.INTER_CUBIC) for i in range(img_m_raw.shape[0])])
	layers[19:20] = np.array([cv2.resize(img_p_raw[i],(size[1],size[0]),interpolation=cv2.INTER_CUBIC) for i in range(img_p_raw.shape[0])])

	#TODO: align different images

	return layers

def load_align_image_layers(image_id, size = (335, 340)):
	img_3_raw = tiff.imread("data/three_band/{}.tif".format(image_id))
	img_a_raw = tiff.imread("data/sixteen_band/{}_A.tif".format(image_id))
	img_m_raw = tiff.imread("data/sixteen_band/{}_M.tif".format(image_id))
	img_p_raw = np.array([tiff.imread("data/sixteen_band/{}_P.tif".format(image_id))])

	img_3 = np.array([cv2.resize(img_3_raw[i],(size[1],size[0]),interpolation=cv2.INTER_CUBIC) for i in range(img_3_raw.shape[0])], dtype='float32')
	img_a = np.array([cv2.resize(img_a_raw[i],(size[1],size[0]),interpolation=cv2.INTER_CUBIC) for i in range(img_a_raw.shape[0])], dtype='float32')
	img_m = np.array([cv2.resize(img_m_raw[i],(size[1],size[0]),interpolation=cv2.INTER_CUBIC) for i in range(img_m_raw.shape[0])], dtype='float32')
	img_p = np.array([cv2.resize(img_p_raw[i],(size[1],size[0]),interpolation=cv2.INTER_CUBIC) for i in range(img_p_raw.shape[0])], dtype='float32')

	img_a = get_alignment_displacement(img_3, img_a)
	img_m = get_alignment_displacement(img_3, img_m)
	img_p = get_alignment_displacement(img_3, img_p)

	layers = np.zeros((20, size[0], size[1]), dtype='uint16')

	layers[0:3] = img_3
	layers[3:11] = img_a
	layers[11:19] = img_m
	layers[19:20] = img_p

	#TODO: align different images

	return layers


inDir = 'data'
df = pd.read_csv(inDir + '/train_wkt_v4.csv')
gs = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

def load_train_image_labels(image_id, size = (3348, 3403)):

	labels = np.zeros((10, size[0], size[1]), dtype='uint8')
	for i in range(10):
		labels[i, :,:] = export_pixel_mask.generate_mask_for_image_and_class((labels.shape[1],labels.shape[2]),image_id,i+1,gs,df)

	return labels

def plot_everything(imaging_layers, labels):
	img_3 = imaging_layers[0:3]
	img_a = imaging_layers[3:11]
	img_m = imaging_layers[11:19]
	img_p = imaging_layers[19:20]

	print(img_3.shape)
	print(img_a.shape)
	print(img_m.shape)
	print(img_p.shape)

	plt.subplot(4, 10, 1); plt.imshow(img_3[0]); plt.title('3 channel')
	plt.subplot(4, 10, 2); plt.imshow(img_3[1]);
	plt.subplot(4, 10, 3); plt.imshow(img_3[2]);
	plt.subplot(4, 10, 4); plt.imshow(np.transpose(img_3, (1, 2, 0)) / (2**11)); plt.title('RGB')

	plt.subplot(4, 10, 10); plt.imshow(img_p[0]); plt.title('Panchromatic')

	plt.subplot(4, 10, 11); plt.imshow(img_a[0]); plt.title('a')
	plt.subplot(4, 10, 12); plt.imshow(img_a[1]);
	plt.subplot(4, 10, 13); plt.imshow(img_a[2]);
	plt.subplot(4, 10, 14); plt.imshow(img_a[3]);
	plt.subplot(4, 10, 15); plt.imshow(img_a[4]);
	plt.subplot(4, 10, 16); plt.imshow(img_a[5]);
	plt.subplot(4, 10, 17); plt.imshow(img_a[6]);
	plt.subplot(4, 10, 18); plt.imshow(img_a[7]);

	plt.subplot(4, 10, 21); plt.imshow(img_m[0]); plt.title('m')
	plt.subplot(4, 10, 22); plt.imshow(img_m[1]);
	plt.subplot(4, 10, 23); plt.imshow(img_m[2]);
	plt.subplot(4, 10, 24); plt.imshow(img_m[3]);
	plt.subplot(4, 10, 25); plt.imshow(img_m[4]);
	plt.subplot(4, 10, 26); plt.imshow(img_m[5]);
	plt.subplot(4, 10, 27); plt.imshow(img_m[6]);
	plt.subplot(4, 10, 28); plt.imshow(img_m[7]);


	plt.subplot(4, 10, 31); plt.imshow(labels[0]);
	plt.subplot(4, 10, 32); plt.imshow(labels[1]);
	plt.subplot(4, 10, 33); plt.imshow(labels[2]);
	plt.subplot(4, 10, 34); plt.imshow(labels[3]);
	plt.subplot(4, 10, 35); plt.imshow(labels[4]);
	plt.subplot(4, 10, 36); plt.imshow(labels[5]);
	plt.subplot(4, 10, 37); plt.imshow(labels[6]);
	plt.subplot(4, 10, 38); plt.imshow(labels[7]);
	plt.subplot(4, 10, 39); plt.imshow(labels[8]);
	plt.subplot(4, 10, 40); plt.imshow(labels[9]);

	# plt.tight_layout()
	plt.show()

def plot_everything2(imaging_layers, labels):
	
	for i in range(20):
		plt.subplot(3, 10, 1+i)
		plt.imshow(imaging_layers[i])

	layer_labels = ['Buildings', 'misc manmade', 'road', 'track', 'trees', 'crops', 'waterway', 'standing water', 'vehicle large', 'vehicle small']

	for i in range(10):
		plt.subplot(3, 10, i+21)
		plt.imshow(labels[i], cmap='gray')
		plt.title(layer_labels[i])

	plt.show()

def test(image_id = "6120_2_2", image_size = (3348, 3403)):
	
	# imaging_layers = load_image_layers(image_id, size = image_size)
	imaging_layers = load_align_image_layers(image_id, size=image_size)

	labels = load_train_image_labels(image_id, size = image_size)

	# get_alignment_displacement(imaging_layers[:3].astype('float32'), labels.astype('float32'), plot=True)


	plot_everything2(imaging_layers, labels)

	# tiff.imsave('cache/labels/%s.tif' % image_id, labels)
	# tiff.imsave('cache/images/%s.tif' % image_id, imaging_layers)

def find_label_intersections(image_id = "6120_2_2", image_size = (3348, 3403)):

	filenames = os.listdir('data/three_band')
	image_ids = list(map(lambda s: s.strip('.tif'), filenames))

	for i, image_id in enumerate(image_ids):
		labels = load_train_image_labels(image_id, size = image_size)

		label_sum = np.sum(labels, axis=0)
	# print(labels.shape, label_insection.shape)
		intersection_sum = np.where(label_sum > 1)[0]
		if len(intersection_sum) > 0:

			for i in range(10):
				plt.subplot(3, 5, i+1)
				plt.imshow(labels[i])

			plt.subplot(3, 5, 13)
			plt.imshow(label_sum)

			plt.show()

			print(intersection_sum)

	plt.imshow(label_insection); plt.show()


def create_all_cache_files(image_size = (339, 340), align=True):
	filenames = os.listdir('data/three_band')
	image_ids = list(map(lambda s: s.strip('.tif'), filenames))

	for i, image_id in enumerate(image_ids):
		sys.stdout.write('\b\b\b\b\b\b\b\b\b\b\b\b\b%d / %d' % (i, len(image_ids))); sys.stdout.flush()

		if align:
			imaging_layers = load_align_image_layers(image_id, size=image_size)
		else:
			imaging_layers = load_image_layers(image_id, size = image_size)
	
		labels = load_train_image_labels(image_id, size = image_size)

		prefix = 'align_' if align else ''

		if np.sum(labels) > 0: 
			tiff.imsave(prefix + 'cache/labels/%s.tif' % image_id, labels)	

		tiff.imsave(prefix + 'cache/images/%s.tif' % image_id, imaging_layers)

if __name__ == "__main__":
	# test()
	# create_all_cache_files()
	find_label_intersections()