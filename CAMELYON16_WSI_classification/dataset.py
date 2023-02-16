import os
import sys
import numpy as np
import torch
import torch.utils.data
import random
import torchvision.transforms.functional as TF

import pickle



class Dataset(torch.utils.data.Dataset):
    def __init__(self, feature_dir=None, slide_list_filename=None, num_instances=16):
        self._feature_dir = feature_dir
        self._num_instances = num_instances

        self._slide_ids, self._slide_num_patches, self._slide_labels, self._slide_features = self.read_slide_data(slide_list_filename)

        self._num_slides = len(self._slide_ids)

    @property
    def num_slides(self):
        return self._num_slides


    def __len__(self):
        return self._num_slides

    def read_slide_data(self, slide_list_filename):
        # read resnet50 features
        if slide_list_filename.split('/')[-1][:-4] != 'test':
            features_file = '{}/mDATA_train.pkl'.format(self._feature_dir)
        else:
            features_file = '{}/mDATA_test.pkl'.format(self._feature_dir)

        with open(features_file, 'rb') as f:
            features = pickle.load(f)

        data_arr = np.loadtxt(slide_list_filename, delimiter='\t', comments='#', dtype=str)
        slide_ids = data_arr[:,0]
        labels = np.asarray(data_arr[:,1], dtype=int)

        slide_ids_list = []
        slide_num_patches_list = []
        slide_labels_list = []
        slide_features_list = []
        for i,temp_slide_id in enumerate(slide_ids):
            temp_label = labels[i]

            temp_features = features[temp_slide_id]
            temp_num_patches = len(temp_features)
            if temp_num_patches == 0:
                print('slide {} is excluded from dataset since there is no cropped patches!'.format(temp_slide_id))
                continue

            temp_features_tensor = []
            for patch_info in temp_features:
                patch_features = torch.from_numpy(patch_info['feature'])

                temp_features_tensor.append(patch_features.unsqueeze(0))

            temp_features_tensor = torch.cat(temp_features_tensor, dim=0)


            slide_ids_list.append(temp_slide_id)
            slide_num_patches_list.append(temp_num_patches)
            slide_labels_list.append(temp_label)
            slide_features_list.append(temp_features_tensor)


        return slide_ids_list, slide_num_patches_list, slide_labels_list, slide_features_list
    
    

    def __getitem__(self, idx):
        temp_slide_id = self._slide_ids[idx]
        temp_slide_num_patches = self._slide_num_patches[idx]
        temp_slide_label = self._slide_labels[idx]
        temp_slide_features = self._slide_features[idx]

        
        if self._num_instances < 1.0:
            patch_indices = np.arange(temp_slide_num_patches)
            np.random.shuffle(patch_indices)

            temp_num_instances = int(temp_slide_num_patches*self._num_instances)
            patch_indices = patch_indices[:temp_num_instances]

            features = temp_slide_features[patch_indices]
        else:
            features = temp_slide_features

        # features = temp_slide_features

        slide_label = torch.as_tensor(temp_slide_label, dtype=torch.int64)
        
        return features, slide_label



def custom_collate_fn(batch):
    features_list, slide_label_list = zip(*batch)

    return torch.cat(features_list,dim=0), torch.stack(slide_label_list,dim=0)

def worker_init_fn(id):
    np.random.seed(torch.initial_seed()&0xffffffff)

