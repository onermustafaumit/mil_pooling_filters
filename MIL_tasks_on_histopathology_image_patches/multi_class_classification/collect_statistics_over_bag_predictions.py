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

def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues,
						  current_ax = None):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		# print("Normalized confusion matrix")
	else:
		# print('Confusion matrix, without normalization')
		pass

	cm_normalized = (cm.astype('float') - np.amin(cm)) / (np.amax(cm)-np.amin(cm))

	# print(cm)

	plt.rcParams.update({'font.size':10, 'font.family':'Times New Roman'})
	ax = current_ax
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	# plt.title(title)
	# plt.colorbar()
	tick_marks = np.arange(len(classes))
	ax.set_xticks(tick_marks)
	ax.set_yticks(tick_marks)
	ax.set_xticklabels(classes)
	ax.set_yticklabels(classes)
	ax.set_ylim( (len(classes)-0.5, -0.5) )


	fmt = '.3f' if normalize else 'd'
	thresh = 0.5
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		ax.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",verticalalignment="center",
				 fontsize=10, fontname='Times New Roman',
				 color="white" if cm_normalized[i, j] > thresh else "black")

	ax.set_ylabel('Truth')
	ax.set_xlabel('Predicted')
	
	# divider = make_axes_locatable(ax)
	# cax = divider.append_axes("right", size="5%", pad=0.05)

	# plt.colorbar(im, cax=cax)
	# plt.tight_layout()


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
	f_image_predictions_file_mpp.write('predicted\t')
	f_image_predictions_file_mpp.write('prob_0\t')
	f_image_predictions_file_mpp.write('prob_1\t')
	f_image_predictions_file_mpp.write('prob_2')
	f_image_predictions_file_mpp.write('\n')


bag_ids_list = []
bag_truths_list = []
bag_preds_list = []
bag_probs_list = []
image_truths_list = []
image_preds_list = []
image_probs_mpp_list = []
for i in range(num_images):

	image_id = image_ids_arr[i]
	print('image {}/{}: {}'.format(i+1,num_images,image_id))

	image_data_folder_path = '{}/{}'.format(data_folder_path,image_id)

	test_metrics_filename = '{}/bag_predictions_{}.txt'.format(image_data_folder_path,image_id)

	test_metrics_data = np.loadtxt(test_metrics_filename, delimiter='\t', comments='#', dtype=str)
	bag_ids_data = np.asarray(test_metrics_data[:,0],dtype=str)
	truths_data = np.asarray(test_metrics_data[:,1],dtype=int)
	preds_data = np.asarray(test_metrics_data[:,2],dtype=int)
	probs_data = np.asarray(test_metrics_data[:,3:],dtype=float)

	bag_ids_list.append(bag_ids_data)
	bag_truths_list.append(truths_data)
	bag_preds_list.append(preds_data)
	bag_probs_list.append(probs_data)

	image_truth = truths_data[0]
	image_probs_mpp = np.mean(probs_data,axis=0)
	image_pred = np.argmax(image_probs_mpp)

	image_truths_list.append(image_truth)
	image_preds_list.append(image_pred)
	image_probs_mpp_list.append(image_probs_mpp)

	with open(image_predictions_file_mpp,'a') as f_image_predictions_file_mpp:
		f_image_predictions_file_mpp.write('{}\t'.format(image_id))
		f_image_predictions_file_mpp.write('{:d}\t'.format(image_truth))
		f_image_predictions_file_mpp.write('{:d}\t'.format(image_pred))
		f_image_predictions_file_mpp.write('{:.3f}\t'.format(image_probs_mpp[0]))
		f_image_predictions_file_mpp.write('{:.3f}\t'.format(image_probs_mpp[1]))
		f_image_predictions_file_mpp.write('{:.3f}\n'.format(image_probs_mpp[2]))


bag_ids_arr = np.concatenate(bag_ids_list, axis=0)
print('bag_ids_arr.shape:{}'.format(bag_ids_arr.shape))
bag_truths_arr = np.concatenate(bag_truths_list, axis=0)
print('bag_truths_arr.shape:{}'.format(bag_truths_arr.shape))
bag_preds_arr = np.concatenate(bag_preds_list, axis=0)
print('bag_preds_arr.shape:{}'.format(bag_preds_arr.shape))
bag_probs_arr = np.concatenate(bag_probs_list, axis=0)
print('bag_probs_arr.shape:{}'.format(bag_probs_arr.shape))
image_truths_arr = np.stack(image_truths_list, axis=0)
print('image_truths_arr.shape:{}'.format(image_truths_arr.shape))
image_preds_arr = np.stack(image_preds_list, axis=0)
print('image_preds_arr.shape:{}'.format(image_preds_arr.shape))
image_probs_mpp_arr = np.stack(image_probs_mpp_list, axis=0)
print('image_probs_mpp_arr.shape:{}'.format(image_probs_mpp_arr.shape))


# collect bag level statistics
temp_truth = bag_truths_arr
temp_pred = bag_preds_arr

conf_mat = confusion_matrix(temp_truth, temp_pred, labels=[0,1,2])
conf_mat_filename = '{}/bag_level_cm.txt'.format(data_folder_path)
np.savetxt(conf_mat_filename, conf_mat, fmt='%d', delimiter='\t')

acc = np.sum(conf_mat.diagonal())/np.sum(conf_mat)

bag_level_statistics_filename = '{}/bag_level_statistics.txt'.format(data_folder_path)
with open(bag_level_statistics_filename, 'w') as f_bag_level_statistics_filename:
	f_bag_level_statistics_filename.write('# acc\n')
	f_bag_level_statistics_filename.write('{}\n'.format(acc))

class_names = ['normal','metastases','boundary']

fig, ax = plt.subplots(figsize=(2,2))
plot_confusion_matrix(conf_mat, classes=class_names, normalize=True, title='Confusion matrix', current_ax=ax)
fig_filename = '{}/bag_level_cm_normalized.png'.format(data_folder_path)
fig.savefig(fig_filename, bbox_inches='tight')
plt.close('all')


fig, ax = plt.subplots(figsize=(2,2))
plot_confusion_matrix(conf_mat, classes=class_names, normalize=False, title='Confusion matrix', current_ax=ax)
fig_filename = '{}/bag_level_cm_unnormalized.png'.format(data_folder_path)
fig.savefig(fig_filename, bbox_inches='tight')
plt.close('all')

# collect image level statistics - mpp
temp_truth = image_truths_arr
temp_pred = image_preds_arr

conf_mat = confusion_matrix(temp_truth, temp_pred, labels=[0,1,2])
conf_mat_filename = '{}/image_level_cm_mpp.txt'.format(data_folder_path)
np.savetxt(conf_mat_filename, conf_mat, fmt='%d', delimiter='\t')

fig, ax = plt.subplots(figsize=(2,2))
plot_confusion_matrix(conf_mat, classes=class_names, normalize=True, title='Confusion matrix', current_ax=ax)
fig_filename = '{}/image_level_cm_normalized_mpp.png'.format(data_folder_path)
fig.savefig(fig_filename, bbox_inches='tight')
plt.close('all')

fig, ax = plt.subplots(figsize=(2,2))
plot_confusion_matrix(conf_mat, classes=class_names, normalize=False, title='Confusion matrix', current_ax=ax)
fig_filename = '{}/image_level_cm_unnormalized_mpp.png'.format(data_folder_path)
fig.savefig(fig_filename, bbox_inches='tight')
plt.close('all')

acc = np.sum(conf_mat.diagonal())/np.sum(conf_mat)

image_level_statistics_filename = '{}/image_level_statistics_mpp.txt'.format(data_folder_path)
with open(image_level_statistics_filename, 'w') as f_image_level_statistics_filename:
	f_image_level_statistics_filename.write('# acc\n')
	f_image_level_statistics_filename.write('{:.4f}\n'.format(acc))

