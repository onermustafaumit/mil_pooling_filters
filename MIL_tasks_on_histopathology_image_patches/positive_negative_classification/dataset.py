import numpy as np
from PIL import Image
import os
import sys

import torch
import torch.utils.data
from torchvision import transforms
import torchvision.transforms.functional as TF


class Dataset(torch.utils.data.Dataset):
	def __init__(self, dataset_dir=None, dataset_type='train', patch_size = 32, num_instances=10):
		self._num_instances = num_instances
		self._dataset_dir = dataset_dir
		self._dataset_type = dataset_type
		self._patch_size = patch_size 

		# load data
		img_list_file = '{}/{}_img_list.txt'.format(dataset_dir,dataset_type)
		img_list = np.loadtxt(img_list_file, delimiter='\t', comments='#', dtype='str') #[:2,:]
		self._img_ids_arr = img_list[:,0]
		self._metastasis_ratio_arr = np.asarray(img_list[:,3], dtype=np.float32)
		self._ucc_arr = np.asarray(img_list[:,4], dtype=int)

		img_data_file = '{}/{}_img_data.npy'.format(dataset_dir,dataset_type)
		self._img_data_arr = np.load(img_data_file) #[:2,:]

		self._labels_arr = self.label_fnc(metastasis_ratio_arr=self._metastasis_ratio_arr, ucc_arr=self._ucc_arr)

		self._indices = np.arange(self._labels_arr.shape[0])

		##### make a balanced dataset for training #####
		indices_list = list()

		# get unique label ids and counts
		unique_ids, unique_id_counts = np.unique(self._labels_arr, return_counts=True)
		max_count = np.amax(unique_id_counts)

		for i in range(unique_ids.shape[0]):
			temp_unique_id = unique_ids[i]
			temp_unique_id_count = unique_id_counts[i]

			temp_indices = np.where(self._labels_arr == temp_unique_id)[0]
			if temp_unique_id_count < max_count:
				temp_indices = np.tile(temp_indices, int(max_count//temp_unique_id_count + 1) )
				temp_indices = temp_indices[:max_count]

			indices_list += list(temp_indices)

		# print('len(indices_list):{}'.format(len(indices_list)))
		indices_arr = np.array(indices_list)

		self._indices = self._indices[indices_arr]
		##### make a balanced dataset #####

		self._num_images = self._indices.shape[0]

		self._img_transforms = self.image_transforms()


	@property
	def num_images(self):
		return self._num_images

	def __len__(self):
		return self._num_images

	def label_fnc(self, metastasis_ratio_arr, ucc_arr):
		# 0:negative, 1:positive
		labels_arr = np.zeros(ucc_arr.shape, dtype=np.uint8)
		
		# find positive bags
		ucc2_indices = (ucc_arr == 2)
		ucc1_fully_metastases_indices = (ucc_arr == 1) & (metastasis_ratio_arr>0.5)

		pos_bag_indices = ucc2_indices | ucc1_fully_metastases_indices

		labels_arr[pos_bag_indices] = 1
		
		return labels_arr

	def image_transforms(self):
		if self._dataset_type == 'train':
			img_transforms = transforms.Compose([	
													transforms.RandomCrop(self._patch_size),
													transforms.RandomRotation(180),
													transforms.RandomHorizontalFlip(),
													transforms.RandomVerticalFlip(),
													transforms.ToTensor(),
													self.MyNormalizationTransform(),
													])

		else:
			img_transforms = transforms.Compose([	
													transforms.RandomCrop(self._patch_size),
													transforms.ToTensor(),
													self.MyNormalizationTransform(),
													])

		return img_transforms


	class MyNormalizationTransform(object):
		def __call__(self, input_tensor):
			mean_tensor = torch.mean(input_tensor).view((1,))
			std_tensor = torch.std(input_tensor).view((1,))

			return TF.normalize(input_tensor, mean_tensor, std_tensor)

	def get_sample_data(self, img_arr):
		img = Image.fromarray(img_arr)

		img_tensor_list = list()
		for i in range(self._num_instances):
			img_tensor = self._img_transforms(img)

			img_tensor_list.append(img_tensor)

		return torch.stack(img_tensor_list,dim=0)


	def __getitem__(self, idx):

		temp_index = self._indices[idx]

		temp_img_id = self._img_ids_arr[temp_index]
		temp_label = self._labels_arr[temp_index]
		temp_img_arr = self._img_data_arr[temp_index]

		temp_sample = self.get_sample_data(img_arr = temp_img_arr)

		temp_label = torch.as_tensor(temp_label, dtype=torch.long)

		return temp_sample, temp_label


def custom_collate_fn(batch):
	sample_tensors_list, label_tensors_list = zip(*batch)

	return torch.cat(sample_tensors_list,dim=0), torch.stack(label_tensors_list,dim=0)



