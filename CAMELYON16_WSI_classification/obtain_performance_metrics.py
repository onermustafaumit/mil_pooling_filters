import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_fscore_support
import math
import itertools
from itertools import cycle

# plt.rcParams.update({'font.size':10, 'font.family':'Times New Roman'})
plt.rcParams.update({'font.size':8})

def score_fnc(data_arr1, data_arr2):
	auc = roc_auc_score(data_arr1, data_arr2)
	return auc

def BootStrap(data_arr1, data_arr2, n_bootstraps):

	# initialization by bootstraping
	n_bootstraps = n_bootstraps
	rng_seed = 42  # control reproducibility
	bootstrapped_scores = []
	# print(data_arr2)
	# print(data_arr2)

	rng = np.random.RandomState(rng_seed)
	
	for i in range(n_bootstraps):
		# bootstrap by sampling with replacement on the prediction indices
		indices = rng.randint(0, len(data_arr2), len(data_arr2))

		if len(np.unique(data_arr1[indices])) < 2:
			# We need at least one sample from each class
			# otherwise reject the sample
			#print("We need at least one sample from each class")
			continue
		else:
			score = score_fnc(data_arr1[indices], data_arr2[indices])
			bootstrapped_scores.append(score)
			#print("score: %f" % score)

	sorted_scores = np.array(bootstrapped_scores)
	sorted_scores.sort()
	if len(sorted_scores)==0:
		return 0., 0.
	# Computing the lower and upper bound of the 95% confidence interval
	# You can change the bounds percentiles to 0.025 and 0.975 to get
	# a 95% confidence interval instead.
	#print(sorted_scores)
	#print(len(sorted_scores))
	#print(int(0.025 * len(sorted_scores)))
	#print(int(0.975 * len(sorted_scores)))
	confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
	confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
	# print(confidence_lower)
	# print(confidence_upper)
	# print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper))
	return sorted_scores, confidence_lower, confidence_upper

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


parser = argparse.ArgumentParser(description='Train a CNN to classify image patches')

parser.add_argument('--metrics_file', default='', help='Text file to write metrics', dest='metrics_file')

FLAGS = parser.parse_args()

class_names = ['-ve','+ve']

data_arr = np.loadtxt(FLAGS.metrics_file, delimiter='\t',comments='#',dtype=str)
label_arr = np.asarray(data_arr[:,1],dtype=int)
positive_score_arr = np.asarray(data_arr[:,2],dtype=float)
pred_arr = np.asarray(positive_score_arr>0.5, dtype=int)

conf_mat = confusion_matrix(label_arr, pred_arr, labels=[0,1])
conf_mat_filename = '{}__slide_level_cm.txt'.format(FLAGS.metrics_file[:-4])
np.savetxt(conf_mat_filename, conf_mat, fmt='%d', delimiter='\t')

fig, ax = plt.subplots(figsize=(2,2))
plot_confusion_matrix(conf_mat, classes=class_names, normalize=True, title='Confusion matrix', current_ax=ax)
fig_filename = '{}__slide_level_cm_normalized.png'.format(FLAGS.metrics_file[:-4])
fig.savefig(fig_filename, bbox_inches='tight')
plt.close('all')

fig, ax = plt.subplots(figsize=(2,2))
plot_confusion_matrix(conf_mat, classes=class_names, normalize=False, title='Confusion matrix', current_ax=ax)
fig_filename = '{}__slide_level_cm_unnormalized.png'.format(FLAGS.metrics_file[:-4])
fig.savefig(fig_filename, bbox_inches='tight')
plt.close('all')

acc = np.sum(conf_mat.diagonal())/np.sum(conf_mat)
precision, recall, fscore, support = precision_recall_fscore_support(label_arr, pred_arr, average='binary', labels=[0,1], pos_label=1)

fpr, tpr, th = roc_curve(label_arr, positive_score_arr, pos_label=1)
auroc = auc(fpr, tpr)
# print(auroc)

distance_to_corner = fpr**2 + (1-tpr)**2
min_distance_index = np.argmin(distance_to_corner)
min_distance_th = th[min_distance_index]

print('min_distance_th:{}'.format(min_distance_th))

sorted_scores, auroc_lower, auroc_upper = BootStrap(label_arr, positive_score_arr, n_bootstraps=2000)


# title_text = 'AUROC = {:.3f} (CI: {:.3f} - {:.3f})'.format(auroc, auroc_lower, auroc_upper)
title_text = 'AUROC = {:.3f} ({:.3f} - {:.3f})'.format(auroc, auroc_lower, auroc_upper)
print(title_text)

fig, ax = plt.subplots(figsize=(3,3))
ax.plot(fpr, tpr, lw=2, alpha=1., color='k')
# ax.plot(fpr, tpr, color='k', lw=2)
# ax.plot([0, 1], [0, 1], 'k--', lw=1)



ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xlim((-0.05,1.05))
ax.set_xticks(np.arange(0,1.05,0.2))
# ax.set_xticklabels('')
ax.set_ylim((-0.05,1.05))
ax.set_yticks(np.arange(0,1.05,0.2))
# ax.set_yticklabels('')
ax.set_axisbelow(True)
ax.grid(color='gray') #, linestyle='dashed')
ax.set_title(title_text)
# ax.legend(framealpha=1.)




fig.tight_layout()
# fig.subplots_adjust(left=0.15, bottom=0.12, right=0.98, top=0.98, wspace=0.20 ,hspace=0.20 )
fig_filename = '{}__roc.pdf'.format(FLAGS.metrics_file[:-4])
fig.savefig(fig_filename, dpi=300)

fig_filename = '{}__roc.png'.format(FLAGS.metrics_file[:-4])
fig.savefig(fig_filename, dpi=300)

# plt.show()

plt.close('all')

slide_level_statistics_filename = '{}__slide_level_statistics.txt'.format(FLAGS.metrics_file[:-4])
with open(slide_level_statistics_filename, 'w') as f_slide_level_statistics_filename:
	f_slide_level_statistics_filename.write('# acc\tprecision\trecall\tfscore\tauroc\tauroc_lower\tauroc_upper\n')
	f_slide_level_statistics_filename.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(acc,precision,recall,fscore,auroc,auroc_lower,auroc_upper))

