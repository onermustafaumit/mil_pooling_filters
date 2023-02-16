import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from distribution_pooling_filter import DistributionPoolingFilter


class FeatureExtractor(nn.Module):
    def __init__(self, num_features=64):
        super().__init__()

        self.feature_nn = nn.Sequential(
            nn.Linear(1024, num_features),
            nn.ReLU(),
            )
        
    def forward(self, x):
        feature_vec = self.feature_nn(x)
        
        return feature_vec


class SlideClassifier(nn.Module):
    def __init__(self, num_features=32, num_classes=10):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_features, num_classes)
            )

    def forward(self, x):

        out = self.fc(x)

        return out

class Attention(nn.Module):
    def __init__(self, num_in=32):
        super(Attention, self).__init__()

        self.fc = nn.Sequential(
                nn.Linear(num_in, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
                )

    def forward(self, x):

        out = self.fc(x)
        out = torch.reshape(out,(1,-1,1))
        out = F.softmax(out, dim=1)

        return out

class Attention2(nn.Module):
    def __init__(self, num_in=32):
        super(Attention2, self).__init__()

        self.fc = nn.Sequential(
                nn.Linear(num_in, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
                )

    def forward(self, x):

        out = self.fc(x)
        out = torch.reshape(out,(1,-1,1))
        out = torch.sigmoid(out)

        return out

class Model(nn.Module):

    def __init__(self, num_classes=2, num_features=32):
        super().__init__()

        self._num_classes = num_classes
        self._num_features = num_features
        self._num_bins = 11
        self._sigma = 0.033

        # feature extractor module
        self._feature_extractor = FeatureExtractor(num_features=num_features)


        self._attention = Attention(num_in=num_features)
        self._attention2 = Attention2(num_in=num_features)

        self._distribution_pooling_filter = DistributionPoolingFilter(num_bins=self._num_bins, sigma=self._sigma)

        # slide-level classifier
        self._slide_classifier = SlideClassifier(num_features=num_features*self._num_bins, num_classes=num_classes)

    def forward(self, x):

        # feature extractor
        extracted_features = self._feature_extractor(x)

        attention_values = self._attention(extracted_features)
        attention_values2 = self._attention2(extracted_features)
        
        extracted_features = torch.reshape(extracted_features,(1, -1, self._num_features))

        # distribution pooling
        out = attention_values2*extracted_features
        out = self._distribution_pooling_filter(out, attention_values)
        out = torch.reshape(out,(-1, self._num_features*self._num_bins))

        # slide classifier
        out = self._slide_classifier(out)

        return out
        