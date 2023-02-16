import os
import sys
import numpy as np
import torch
import torch.utils.data
import random
import torchvision.transforms.functional as TF

import pickle



class Dataset(torch.utils.data.Dataset):
    def __init__(self, feature_dir=None, dataset_type='test'):
        self._feature_dir = feature_dir
        self._num_bags = 1

        # read resnet50 features
        if dataset_type != 'test':
            features_file = '{}/mDATA_train.pkl'.format(self._feature_dir)
        else:
            features_file = '{}/mDATA_test.pkl'.format(self._feature_dir)

        with open(features_file, 'rb') as f:
            self._features = pickle.load(f)


    @property
    def num_patches(self):
        return self._slide_num_patches

    def __len__(self):
        return self._num_bags

    def update_slide_id(self, slide_id):
        self._slide_id = slide_id

        self._slide_num_patches, self._slide_features = self.read_slide_data()



    def read_slide_data(self):
        
        temp_features = self._features[self._slide_id]
        temp_num_patches = len(temp_features)
        if temp_num_patches == 0:
            print('slide {} is excluded from dataset since there is no cropped patches!'.format(temp_slide_id))

        temp_features_tensor = []
        for patch_info in temp_features:
            patch_features = torch.from_numpy(patch_info['feature'])

            temp_features_tensor.append(patch_features.unsqueeze(0))

        temp_features_tensor = torch.cat(temp_features_tensor, dim=0)

        return temp_num_patches, temp_features_tensor
    
    

    def __getitem__(self, idx):

        features = self._slide_features
        
        return features, 0


def custom_collate_fn(batch):
    features_list, slide_label_list = zip(*batch)

    # return torch.cat(features_list,dim=0), torch.stack(slide_label_list,dim=0)
    return torch.cat(features_list,dim=0)

def worker_init_fn(id):
    np.random.seed(torch.initial_seed()&0xffffffff)

