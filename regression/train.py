import numpy as np
import argparse
from datetime import datetime
import os
import sys
import time

from model import Model
from dataset import Dataset, custom_collate_fn

import torch
import torch.utils.data

from tqdm import tqdm

parser = argparse.ArgumentParser(description='')

parser.add_argument('--model_dir', default='saved_models', help='Directory to save models', dest='model_dir')
parser.add_argument('--init_model_file', default='', help='Initial model file (optional)', dest='init_model_file')
parser.add_argument('--dataset_dir', default='../dataset', help='', dest='dataset_dir')
parser.add_argument('--patch_size', default='32', type=int, help='Patch size', dest='patch_size')
parser.add_argument('--num_instances', default='64', type=int, help='Number of instances', dest='num_instances')
parser.add_argument('--num_classes', default='1', type=int, help='Number of classes', dest='num_classes')
parser.add_argument('--batch_size', default='32', type=int, help='Batch size', dest='batch_size')
parser.add_argument('--mil_pooling_filter', default='distribution_general', help='MIL pooling filter: distribution_general, mean, attention, max', dest='mil_pooling_filter')
parser.add_argument('--num_bins', default='21', type=int, help='Number of bins in distribution pooling filters', dest='num_bins')
parser.add_argument('--sigma', default='0.0167', type=float, help='sigma in Gaussian kernel in distribution pooling filters', dest='sigma')
parser.add_argument('--num_features', default='32', type=int, help='Number of features', dest='num_features')
parser.add_argument('--learning_rate', default='1e-4', type=float, help='Learning rate', dest='learning_rate')
parser.add_argument('--num_epochs', default=1000, type=int, help='Number of epochs', dest='num_epochs')
parser.add_argument('--save_interval', default=50, type=int, help='Model save interval (default: 1000)', dest='save_interval')
parser.add_argument('--metrics_file', default='loss_data', help='Text file to write step, loss, accuracy metrics', dest='metrics_file')

FLAGS = parser.parse_args()

current_time = datetime.now().strftime("__%Y_%m_%d__%H_%M_%S")
metrics_file = '{}/step_loss_acc_metrics{}.txt'.format(FLAGS.metrics_file, current_time)

print('Model parameters:')
print('dataset_dir = {}'.format(FLAGS.dataset_dir))
print('num_classes = {}'.format(FLAGS.num_classes))
print('num_features = {}'.format(FLAGS.num_features))
print('num_instances = {}'.format(FLAGS.num_instances))
print('batch_size = {}'.format(FLAGS.batch_size))
print('mil_pooling_filter = {}'.format(FLAGS.mil_pooling_filter))
print('num_bins = {}'.format(FLAGS.num_bins))
print('sigma = {}'.format(FLAGS.sigma))
print('learning_rate = {}'.format(FLAGS.learning_rate))
print('num_epochs = {}'.format(FLAGS.num_epochs))
print('metrics_file = {}'.format(FLAGS.metrics_file))

train_dataset = Dataset(dataset_dir=FLAGS.dataset_dir, 
						dataset_type='train', 
						patch_size = FLAGS.patch_size,
						num_instances=FLAGS.num_instances)
num_images_train = train_dataset.num_images
print("Training Data - num_images: {}".format(num_images_train))

val_dataset = Dataset(dataset_dir=FLAGS.dataset_dir, 
						dataset_type='val', 
						patch_size = FLAGS.patch_size,
						num_instances=FLAGS.num_instances)
num_images_val = val_dataset.num_images
print("Validation Data - num_images: {}".format(num_images_val))

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=1, collate_fn=custom_collate_fn)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=1, collate_fn=custom_collate_fn)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(num_classes=FLAGS.num_classes, 
				num_instances=FLAGS.num_instances, 
				num_features=FLAGS.num_features, 
				mil_pooling_filter=FLAGS.mil_pooling_filter,
				num_bins=FLAGS.num_bins, 
				sigma=FLAGS.sigma)
model.to(device)

if FLAGS.init_model_file:
	if os.path.isfile(FLAGS.init_model_file):
		state_dict = torch.load(FLAGS.init_model_file)
		model.load_state_dict(state_dict['model_state_dict'])
		optimizer.load_state_dict(state_dict['optimizer_state_dict'])
		print('weights loaded successfully!!!\n{}'.format(FLAGS.init_model_file))

# define loss criterion
criterion = torch.nn.L1Loss()

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=FLAGS.learning_rate, weight_decay=0.0005)

with open(metrics_file,'w') as f_metric_file:
	f_metric_file.write('# Model parameters:\n')
	f_metric_file.write('# dataset_dir = {}\n'.format(FLAGS.dataset_dir))
	f_metric_file.write('# patch_size = {}\n'.format(FLAGS.patch_size))
	f_metric_file.write('# num_instances = {}\n'.format(FLAGS.num_instances))
	f_metric_file.write('# num_classes = {}\n'.format(FLAGS.num_classes))
	f_metric_file.write('# batch_size = {}\n'.format(FLAGS.batch_size))
	f_metric_file.write('# learning_rate = {}\n'.format(FLAGS.learning_rate))
	f_metric_file.write("# Training Data - num_images: {}\n".format(num_images_train))
	f_metric_file.write("# Validation Data - num_images: {}\n".format(num_images_val))
	f_metric_file.write('# mil_pooling_filter = {}\n'.format(FLAGS.mil_pooling_filter))
	f_metric_file.write('# num_bins = {}\n'.format(FLAGS.num_bins))
	f_metric_file.write('# sigma = {}\n'.format(FLAGS.sigma))
	f_metric_file.write('# num_features = {}\n'.format(FLAGS.num_features))
	f_metric_file.write('# metrics_file = {}\n'.format(FLAGS.metrics_file))
	f_metric_file.write('# model_dir: {}\n'.format(FLAGS.model_dir))
	f_metric_file.write('# init_model_file: {}\n'.format(FLAGS.init_model_file))
	f_metric_file.write('# num_epochs: {}\n'.format(FLAGS.num_epochs))
	f_metric_file.write('# save_interval = {}\n'.format(FLAGS.save_interval))
	f_metric_file.write('# epoch\ttraining_loss\tvalidation_loss\n')

for epoch in range(FLAGS.num_epochs):
	# print('############## EPOCH - {} ##############'.format(epoch+1))
	training_loss = 0
	validation_loss = 0

	# train for one epoch
	# print('******** training ********')
	
	num_predictions = 0

	pbar = tqdm(total=len(train_data_loader))
	
	model.train()
	for images, targets in train_data_loader:
		# print(images.size())
		# print(targets.size())
		images = images.to(device)
		targets = targets.to(device)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		y_logits = model(images)
		loss = criterion(y_logits, targets)
		loss.backward()
		optimizer.step()

		training_loss += loss.item()*targets.size(0)

		num_predictions += targets.size(0)

		pbar.update(1)

		# print(loss.item())

	training_loss /= num_predictions

	pbar.close()


	# evaluate on the validation dataset
	# print('******** validation ********')

	num_predictions = 0

	pbar = tqdm(total=len(val_data_loader))

	model.eval()
	with torch.no_grad():
		for images, targets in val_data_loader:
			images = images.to(device)
			targets = targets.to(device)

			# forward
			y_logits = model(images)
			loss = criterion(y_logits, targets)

			validation_loss += loss.item()*targets.size(0)

			num_predictions += targets.size(0)

			pbar.update(1)

	validation_loss /= num_predictions

	pbar.close()

	print('Epoch=%d ### training_loss=%5.3f ### validation_loss=%5.3f' % (epoch+1, training_loss, validation_loss))

	with open(metrics_file,'a') as f_metric_file:
		f_metric_file.write('%d\t%5.3f\t%5.3f\n' % (epoch+1, training_loss, validation_loss))

	# save model
	if (epoch+1) % FLAGS.save_interval == 0:
		model_weights_filename = FLAGS.model_dir + "/state_dict" + current_time + '__' + str(epoch+1) + ".pth"
		state_dict = {	'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict()}
		torch.save(state_dict, model_weights_filename)
		print("Model weights saved in file: ", model_weights_filename)


print('Training finished!!!')

model_weights_filename = FLAGS.model_dir + "/state_dict" + current_time + '__' + str(epoch+1) + ".pth"
state_dict = {	'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}
torch.save(state_dict, model_weights_filename)
print("Model weights saved in file: ", model_weights_filename)

