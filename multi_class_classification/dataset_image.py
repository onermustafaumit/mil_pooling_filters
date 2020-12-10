import numpy as np
from PIL import Image
import os
import sys

import torch
import torch.utils.data
from torchvision import transforms
import torchvision.transforms.functional as TF


class Dataset(torch.utils.data.Dataset):
	def __init__(self, dataset_dir=None, dataset_type='test', patch_size = 32, num_instances=10, num_bags_per_image=100):
		self._num_instances = num_instances
		self._dataset_dir = dataset_dir
		self._dataset_type = dataset_type
		self._patch_size = patch_size
		self._num_bags_per_image = num_bags_per_image

		# load data
		img_list_file = '{}/{}_img_list.txt'.format(dataset_dir,dataset_type)
		img_list = np.loadtxt(img_list_file, delimiter='\t', comments='#', dtype='str')
		self._img_ids_arr = img_list[:,0]
		self._metastasis_ratio_arr = np.asarray(img_list[:,3], dtype=np.float32)
		self._ucc_arr = np.asarray(img_list[:,4], dtype=int)

		img_data_file = '{}/{}_img_data.npy'.format(dataset_dir,dataset_type)
		self._img_data_arr = np.load(img_data_file)

		self._labels_arr = self.label_fnc(metastasis_ratio_arr=self._metastasis_ratio_arr, ucc_arr=self._ucc_arr)

		self._indices = np.arange(self._labels_arr.shape[0])

		self._num_images = self._indices.shape[0]

		self._img_transforms = self.image_transforms()

		self._current_image_ind = -1


	@property
	def num_images(self):
		return self._num_images

	@property
	def image_ids_arr(self):
		return self._img_ids_arr

	def __len__(self):
		return self._num_bags_per_image

	def label_fnc(self, metastasis_ratio_arr, ucc_arr):
		# 0:normal, 1:metastases, 2:boundary
		labels_arr = np.zeros(ucc_arr.shape, dtype=np.uint8)
		
		# find metastases images
		ucc1_fully_metastases_indices = (ucc_arr == 1) & (metastasis_ratio_arr>0.5)
		labels_arr[ucc1_fully_metastases_indices] = 1

		# find boundary images
		ucc2_indices = (ucc_arr == 2)
		labels_arr[ucc2_indices] = 2
		
		return labels_arr

	def next_image(self):

		self._current_image_ind += 1

		if self._current_image_ind >= self._num_images:
			print('Error: current image index exceeded range!!!')
			sys.exit()

	def image_transforms(self):
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

		temp_index = self._current_image_ind

		temp_img_id = self._img_ids_arr[temp_index]
		temp_label = self._labels_arr[temp_index]
		temp_img_arr = self._img_data_arr[temp_index]

		temp_sample = self.get_sample_data(img_arr = temp_img_arr)

		temp_label = torch.as_tensor(temp_label, dtype=torch.long)

		return temp_sample, temp_label




def custom_collate_fn(batch):
	sample_tensors_list, label_tensors_list = zip(*batch)

	return torch.cat(sample_tensors_list,dim=0), torch.stack(label_tensors_list,dim=0)



