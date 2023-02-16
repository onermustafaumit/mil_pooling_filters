import argparse
from datetime import datetime
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from model import Model
from dataset_slide import Dataset, custom_collate_fn, worker_init_fn
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Train a CNN to classify image patches')

parser.add_argument('--init_model_file', default='',help='Initial model file (optional)', dest='init_model_file')
parser.add_argument('--feature_dir', default='./resnet50_features', help='Image directory', dest='feature_dir')
parser.add_argument('--slide_list_filename', default='./dataset/test.txt', help='slide list test', dest='slide_list_filename')
parser.add_argument('--num_features', default='32', type=int, help='number of features', dest='num_features')
parser.add_argument('--num_classes', default='2', type=int, help='Number of classes', dest='num_classes')
parser.add_argument('--batch_size', default='1', type=int, help='Batch size', dest='batch_size')
parser.add_argument('--metrics_dir', default='test_metrics/', help='Text file to write metrics', dest='metrics_dir')

FLAGS = parser.parse_args()
    
model_name = FLAGS.init_model_file.split('/')[-1][15:-4]

out_dir = '{}/{}/{}'.format(FLAGS.metrics_dir,model_name,FLAGS.slide_list_filename.split('/')[-1][:-4])
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print('init_model_file: {}'.format(FLAGS.init_model_file))
print('feature_dir: {}'.format(FLAGS.feature_dir))
print('slide_list_filename: {}'.format(FLAGS.slide_list_filename))
print('num_features: {}'.format(FLAGS.num_features))
print('num_classes: {}'.format(FLAGS.num_classes))
print('batch_size: {}'.format(FLAGS.batch_size))
print('metrics_dir: {}'.format(FLAGS.metrics_dir))


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# get the model using helper function
model = Model(num_classes=FLAGS.num_classes, num_features=FLAGS.num_features)
# move model to the right device
model.to(device)

if FLAGS.init_model_file:
    if os.path.isfile(FLAGS.init_model_file):
        state_dict = torch.load(FLAGS.init_model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model_state_dict'])
        print("Model weights loaded successfully from file: ", FLAGS.init_model_file)
    else:
        raise Exception("Given model weights file cannot be found!")
else:
    raise Exception("No model weights file is given!")


# read slide list
data_arr = np.loadtxt(FLAGS.slide_list_filename, delimiter='\t', comments='#', dtype=str)
slide_ids = data_arr[:,0]
labels = np.asarray(data_arr[:,1], dtype=int)
num_slides = slide_ids.shape[0]
print('num_slides:{}'.format(num_slides))

# dataset for the current slide
dataset_type = FLAGS.slide_list_filename.split('/')[-1][:-4]
dataset = Dataset(feature_dir=FLAGS.feature_dir, dataset_type=dataset_type)

# define data loader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=5, collate_fn=custom_collate_fn, worker_init_fn=worker_init_fn)


model.eval()
with torch.no_grad():

    for s, slide_id in enumerate(slide_ids):
        print('slide {}/{}: {}'.format(s+1,num_slides,slide_id))

        slide_label = labels[s]

        dataset.update_slide_id(slide_id)

        metrics_file = '{}/test_scores__{}.txt'.format(out_dir,slide_id)
        with open(metrics_file, 'w') as f:
            f.write('# init_model_file: {}\n'.format(FLAGS.init_model_file))
            f.write('# model_name: {}\n'.format(model_name))
            f.write('# slide_list_filename: {}\n'.format(FLAGS.slide_list_filename))
            f.write('# feature_dir: {}\n'.format(FLAGS.feature_dir))
            f.write('# num_features: {}\n'.format(FLAGS.num_features))
            f.write('# num_classes: {}\n'.format(FLAGS.num_classes))
            f.write('# batch_size: {}\n'.format(FLAGS.batch_size))
            f.write('# slide_id\tbag_id\tslide_label\tprediction\tscore_negative\tscore_positive\n')

        if dataset.num_patches == 0:
            for idx in range(FLAGS.num_bags):
                with open(metrics_file, 'a') as f:
                    f.write('{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\n'.format(slide_id, idx, slide_label, 0, 1.0, 0.0))

            continue


        bag_count = 0
        pbar = tqdm(total=len(data_loader))
        for i,img in enumerate(data_loader):
            # print(img.shape)
            img = img.to(device)

            # get logits from the model
            output = model(img)

            # obtain probs
            probs = F.softmax(output, dim=1)

            # obtain predictions
            _, predicted = torch.max(output, 1)

            predicted_arr = predicted.cpu().numpy()
            probs_arr = probs.cpu().numpy()

            temp_num_predictions = predicted_arr.shape[0]
            for idx in range(temp_num_predictions):
                with open(metrics_file, 'a') as f:
                    f.write('{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\n'.format(slide_id, bag_count, slide_label, predicted_arr[idx], probs_arr[idx, 0], probs_arr[idx, 1]))

                bag_count += 1 

            pbar.update(1)

        pbar.close()
