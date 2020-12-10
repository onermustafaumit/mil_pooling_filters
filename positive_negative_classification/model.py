import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from resnet_no_bn import resnet18

from distribution_pooling_filter import DistributionPoolingFilter, DistributionWithAttentionPoolingFilter

class FeatureExtractor(nn.Module):

	def __init__(self, num_out=32):
		super(FeatureExtractor, self).__init__()

		self._model_conv = resnet18()
		
		num_ftrs = self._model_conv.fc.in_features
		self._model_conv.fc = nn.Linear(num_ftrs, num_out)
		# print(self._model_conv)

		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		out = self._model_conv(x)
		out = self.sigmoid(out)

		return out

class RepresentationTransformation(nn.Module):
	def __init__(self, num_in=32, num_out=10):
		super(RepresentationTransformation, self).__init__()

		self.fc = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(num_in, 128),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(128, 32),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(32, num_out)
			)

	def forward(self, x):

		out = self.fc(x)

		return out

class Attention(nn.Module):
	def __init__(self, num_in=32, num_instances=32):
		super(Attention, self).__init__()
		self._num_instances = num_instances

		self.fc = nn.Sequential(
				nn.Linear(num_in, 128),
				nn.Tanh(),
				nn.Linear(128, 1)
				)

	def forward(self, x):

		out = self.fc(x)
		out = torch.reshape(out,(-1,self._num_instances,1))
		out = F.softmax(out, dim=1)

		return out

class Attention2(nn.Module):
	def __init__(self, num_in=32, num_instances=32):
		super(Attention2, self).__init__()
		self._num_instances = num_instances

		self.fc = nn.Sequential(
				nn.Linear(num_in, 128),
				nn.ReLU(),
				nn.Linear(128, 1)
				)

	def forward(self, x):

		out = self.fc(x)
		out = torch.reshape(out,(-1,self._num_instances,1))
		out = torch.sigmoid(out)

		return out


class Model(nn.Module):

	def __init__(self, num_classes=10, num_instances=32, num_features=32, mil_pooling_filter='distribution', num_bins=11, sigma=0.1):
		super(Model, self).__init__()
		self._num_classes = num_classes
		self._num_instances = num_instances
		self._num_features = num_features
		self._num_bins = num_bins
		self._sigma = sigma
		self._mil_pooling_filter = mil_pooling_filter

		self._feature_extractor = FeatureExtractor(num_out=num_features)

		if mil_pooling_filter in ['attention', 'distribution_with_attention', 'distribution_general']:
			self._attention = Attention(num_in=num_features, num_instances=num_instances)

		if mil_pooling_filter == 'distribution_general':
			self._attention2 = Attention2(num_in=num_features, num_instances=num_instances)


		if mil_pooling_filter in ['distribution', 'distribution_with_attention', 'distribution_general']:
			if mil_pooling_filter == 'distribution':
				self._distribution_pooling_filter = DistributionPoolingFilter(num_bins=num_bins, sigma=sigma)
			else:
				self._distribution_with_attention_pooling_filter = DistributionWithAttentionPoolingFilter(num_bins=num_bins, sigma=sigma)

			self._representation_transformation = RepresentationTransformation(num_in=num_features*num_bins, num_out=num_classes)
		else:
			self._representation_transformation = RepresentationTransformation(num_in=num_features, num_out=num_classes)

		# initialize weights
		for m in self.modules():
			if isinstance(m, (nn.Conv2d,nn.Linear)):
				nn.init.xavier_uniform_(m.weight)

	def forward(self, x):

		extracted_features = self._feature_extractor(x)

		if self._mil_pooling_filter in ['attention', 'distribution_with_attention', 'distribution_general']:
			attention_values = self._attention(extracted_features)

			if self._mil_pooling_filter == 'distribution_general':
				attention_values2 = self._attention2(extracted_features)	

		extracted_features = torch.reshape(extracted_features,(-1,self._num_instances,self._num_features))

		if self._mil_pooling_filter == 'distribution':
			out = self._distribution_pooling_filter(extracted_features)
			out = torch.reshape(out,(-1, self._num_features*self._num_bins))

		elif self._mil_pooling_filter == 'distribution_with_attention':
			out = self._distribution_with_attention_pooling_filter(extracted_features, attention_values)
			out = torch.reshape(out,(-1, self._num_features*self._num_bins))
		
		elif self._mil_pooling_filter == 'distribution_general':
			out = attention_values2*extracted_features
			out = self._distribution_with_attention_pooling_filter(out, attention_values)
			out = torch.reshape(out,(-1, self._num_features*self._num_bins))

		elif self._mil_pooling_filter == 'mean':
			out = torch.mean(extracted_features, dim=1)
		
		elif self._mil_pooling_filter == 'attention':
			out = torch.matmul(torch.transpose(attention_values, 2, 1),extracted_features)
			out = torch.squeeze(out, dim=1)

		elif self._mil_pooling_filter == 'max':
			out = torch.max(extracted_features, dim=1)[0]


		out = self._representation_transformation(out)

		return out




		


