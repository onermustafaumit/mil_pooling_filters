import numpy as np
import argparse
import os
import sys
import time

from model import Model
from dataset_image import Dataset, custom_collate_fn

import torch
import torch.utils.data
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='')

parser.add_argument('--model_dir', default='saved_models', help='Directory to save models', dest='model_dir')
parser.add_argument('--init_model_file', default='', help='Initial model file (optional)', dest='init_model_file')
parser.add_argument('--dataset_dir', default='../dataset', help='Dataset dir', dest='dataset_dir')
parser.add_argument('--patch_size', default='32', type=int, help='Patch size', dest='patch_size')
parser.add_argument('--num_instances', default='64', type=int, help='Number of instances', dest='num_instances')
parser.add_argument('--num_classes', default='2', type=int, help='Number of classes', dest='num_classes')
parser.add_argument('--batch_size', default='100', type=int, help='Batch size', dest='batch_size')
parser.add_argument('--mil_pooling_filter', default='distribution_general', help='MIL pooling filter: distribution_general, mean, attention, max', dest='mil_pooling_filter')
parser.add_argument('--num_bins', default='21', type=int, help='Number of bins in distribution pooling filters', dest='num_bins')
parser.add_argument('--sigma', default='0.0167', type=float, help='sigma in Gaussian kernel in distribution pooling filters', dest='sigma')
parser.add_argument('--num_features', default='32', type=int, help='Number of features', dest='num_features')
parser.add_argument('--test_metrics_dir', default='test_metrics', help='Text file to write test metrics', dest='test_metrics_dir')
parser.add_argument('--num_bags_per_image', default='100', type=int, help='number of bags per image to be tested', dest='num_bags_per_image')
parser.add_argument('--dataset_type', default='test', help='', dest='dataset_type')
FLAGS = parser.parse_args()

dataset = Dataset(dataset_dir=FLAGS.dataset_dir, 
					dataset_type=FLAGS.dataset_type, 
					patch_size = FLAGS.patch_size,
					num_instances=FLAGS.num_instances, 
					num_bags_per_image= FLAGS.num_bags_per_image)
num_images = dataset.num_images
image_ids_arr = dataset.image_ids_arr
print("Data - num_images: {}".format(num_images))

data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=1, collate_fn=custom_collate_fn)

model_name = FLAGS.init_model_file.split('__')[1] + '__' + FLAGS.init_model_file.split('__')[2] + '__' + FLAGS.init_model_file.split('__')[3][:-4]
data_folder_path = '{}/{}/{}'.format(FLAGS.test_metrics_dir,model_name,FLAGS.dataset_type)
if not os.path.exists(data_folder_path):
	os.makedirs(data_folder_path)


print('Model parameters:')
print('dataset_dir = {}'.format(FLAGS.dataset_dir))
print('dataset_type = {}'.format(FLAGS.dataset_type))
print('num_classes = {}'.format(FLAGS.num_classes))
print('num_features = {}'.format(FLAGS.num_features))
print('num_instances = {}'.format(FLAGS.num_instances))
print('patch_size = {}'.format(FLAGS.patch_size))
print('batch_size = {}'.format(FLAGS.batch_size))
print('mil_pooling_filter = {}'.format(FLAGS.mil_pooling_filter))
print('num_bins = {}'.format(FLAGS.num_bins))
print('sigma = {}'.format(FLAGS.sigma))
print('init_model_file = {}'.format(FLAGS.init_model_file))

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(num_classes=FLAGS.num_classes, 
				num_instances=FLAGS.num_instances, 
				num_features=FLAGS.num_features, 
				mil_pooling_filter=FLAGS.mil_pooling_filter,
				num_bins=FLAGS.num_bins, 
				sigma=FLAGS.sigma)
model.to(device)

state_dict = torch.load(FLAGS.init_model_file)
model.load_state_dict(state_dict['model_state_dict'])
print('weights loaded successfully!!!\n{}'.format(FLAGS.init_model_file))

model.eval()
with torch.no_grad():

	# num_images = 2 # for sanity check
	for i in range(num_images):
		dataset.next_image()

		image_id = image_ids_arr[i][:-4]
		print('image {}/{}: {}'.format(i+1,num_images,image_id))

		image_data_folder_path = '{}/{}'.format(data_folder_path,image_id)
		if not os.path.exists(image_data_folder_path):
			os.makedirs(image_data_folder_path)

		test_metrics_filename = '{}/bag_predictions_{}.txt'.format(image_data_folder_path,image_id)

		with open(test_metrics_filename,'w') as f_metric_file:
			f_metric_file.write('# Model parameters:\n')
			f_metric_file.write('# dataset_dir = {}\n'.format(FLAGS.dataset_dir))
			f_metric_file.write('# dataset_type = {}\n'.format(FLAGS.dataset_type))
			f_metric_file.write('# patch_size = {}\n'.format(FLAGS.patch_size))
			f_metric_file.write('# num_instances = {}\n'.format(FLAGS.num_instances))
			f_metric_file.write('# num_classes = {}\n'.format(FLAGS.num_classes))
			f_metric_file.write('# mil_pooling_filter = {}\n'.format(FLAGS.mil_pooling_filter))
			f_metric_file.write('# num_bins = {}\n'.format(FLAGS.num_bins))
			f_metric_file.write('# sigma = {}\n'.format(FLAGS.sigma))
			f_metric_file.write('# num_features = {}\n'.format(FLAGS.num_features))
			f_metric_file.write('# init_model_file: {}\n'.format(FLAGS.init_model_file))
			f_metric_file.write('# bag_id\t')
			f_metric_file.write('truth\t')
			f_metric_file.write('predicted')
			for i in range(FLAGS.num_classes):
				f_metric_file.write('\tprob_{}'.format(i))
			f_metric_file.write('\n')


		bag_id = 0 
		for images, targets in data_loader:
			images = images.to(device)
			# print(images.size())
			# print(targets.size())
			
			# get logits from model
			batch_logits = model(images)
			batch_probs = F.softmax(batch_logits, dim=1)
			batch_probs_arr = batch_probs.cpu().numpy()

			num_samples = targets.size(0)
			# print('num_samples: {}'.format(num_samples))

			batch_truths = np.asarray(targets.numpy(),dtype=int)
			batch_predicteds = np.argmax(batch_probs_arr, axis=1)



			with open(test_metrics_filename,'a') as f_metric_file:
				for b in range(num_samples):
					f_metric_file.write('{}_{}\t'.format(image_id,bag_id))
					f_metric_file.write('{:d}\t'.format(batch_truths[b]))
					f_metric_file.write('{:d}'.format(batch_predicteds[b]))
					for c in range(FLAGS.num_classes):
						f_metric_file.write('\t{:.3f}'.format(batch_probs_arr[b,c]))
					f_metric_file.write('\n')

					bag_id += 1

print('Test finished!!!')



