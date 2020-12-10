import numpy as np
import argparse
import os
import sys
from os import path
import itertools
from itertools import cycle

from sklearn.metrics import confusion_matrix, roc_curve, auc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import stats

parser = argparse.ArgumentParser(description='')

parser.add_argument('--data_folder_path', default='', help='data folder path', dest='data_folder_path')

FLAGS = parser.parse_args()

data_folder_path = FLAGS.data_folder_path

image_ids_list = [d for d in os.listdir(data_folder_path) if os.path.isdir(os.path.join(data_folder_path, d))]
# image_ids_list = os.listdir(data_folder_path)

image_ids_arr = np.asarray(sorted(image_ids_list))
num_images = image_ids_arr.shape[0]
print("Data - num_images: {}".format(num_images))

# initiate image predictions files: mpp-mean patch prob
image_predictions_file_mpp = '{}/image_predictions_mpp.txt'.format(data_folder_path)
with open(image_predictions_file_mpp,'w') as f_image_predictions_file_mpp:
	f_image_predictions_file_mpp.write('# image_id\t')
	f_image_predictions_file_mpp.write('truth\t')
	f_image_predictions_file_mpp.write('predicted\n')


bag_ids_list = []
bag_truths_list = []
bag_preds_list = []
image_truths_list = []
image_preds_list = []
for i in range(num_images):

	image_id = image_ids_arr[i]
	print('image {}/{}: {}'.format(i+1,num_images,image_id))

	image_data_folder_path = '{}/{}'.format(data_folder_path,image_id)

	test_metrics_filename = '{}/bag_predictions_{}.txt'.format(image_data_folder_path,image_id)

	test_metrics_data = np.loadtxt(test_metrics_filename, delimiter='\t', comments='#', dtype=str)
	bag_ids_data = np.asarray(test_metrics_data[:,0],dtype=str)
	truths_data = np.asarray(test_metrics_data[:,1],dtype=float)
	preds_data = np.asarray(test_metrics_data[:,2],dtype=float)

	bag_ids_list.append(bag_ids_data)
	bag_truths_list.append(truths_data)
	bag_preds_list.append(preds_data)

	image_truth = truths_data[0]
	image_pred = np.mean(preds_data,axis=0)

	image_truths_list.append(image_truth)
	image_preds_list.append(image_pred)

	with open(image_predictions_file_mpp,'a') as f_image_predictions_file_mpp:
		f_image_predictions_file_mpp.write('{}\t'.format(image_id))
		f_image_predictions_file_mpp.write('{:.4f}\t'.format(image_truth))
		f_image_predictions_file_mpp.write('{:.4f}\n'.format(image_pred))


bag_ids_arr = np.concatenate(bag_ids_list, axis=0)
print('bag_ids_arr.shape:{}'.format(bag_ids_arr.shape))
bag_truths_arr = np.concatenate(bag_truths_list, axis=0)
print('bag_truths_arr.shape:{}'.format(bag_truths_arr.shape))
bag_preds_arr = np.concatenate(bag_preds_list, axis=0)
print('bag_preds_arr.shape:{}'.format(bag_preds_arr.shape))
image_truths_arr = np.stack(image_truths_list, axis=0)
print('image_truths_arr.shape:{}'.format(image_truths_arr.shape))
image_preds_arr = np.stack(image_preds_list, axis=0)
print('image_preds_arr.shape:{}'.format(image_preds_arr.shape))


# collect bag level statistics
temp_truth = bag_truths_arr
temp_pred = bag_preds_arr

abs_error = np.mean(np.abs(temp_truth - temp_pred))

bag_level_statistics_filename = '{}/bag_level_statistics.txt'.format(data_folder_path)
with open(bag_level_statistics_filename, 'w') as f_bag_level_statistics_filename:
	f_bag_level_statistics_filename.write('# abs_error\n')
	f_bag_level_statistics_filename.write('{}\n'.format(abs_error))

# collect image level statistics - mpp
temp_truth = image_truths_arr
temp_pred = image_preds_arr

abs_error = np.mean(np.abs(temp_truth - temp_pred))

image_level_statistics_filename = '{}/image_level_statistics_mpp.txt'.format(data_folder_path)
with open(image_level_statistics_filename, 'w') as f_image_level_statistics_filename:
	f_image_level_statistics_filename.write('# abs_error\n')
	f_image_level_statistics_filename.write('{:.4f}\n'.format(abs_error))

