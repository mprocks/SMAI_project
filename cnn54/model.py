import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
Model for the character-based CNN having 54 Convolutional layers, as follows:
	1 Conv layer from embedding layer (70 characters) to 64 channels

	20 Conv layers from 64 to 64 channels followed by Batch Normalization layers
	1 Pooling layer to reduce the size by 2

	1 Conv layers from 64 to 128 channels followed by a Batch Normalization layer
	18 Conv layers from 128 to 128 channels followed by Batch Normalization layers
	1 Pooling layer to reduce the size by 2

	1 Conv layers from 128 to 256 channels followed by a Batch Normalization layer
	6 Conv layers from 256 to 256 channels followed by Batch Normalization layers
	1 Pooling layer to reduce the size by 2

	1 Conv layers from 256 to 512 channels followed by a Batch Normalization layer
	6 Conv layers from 512 to 512 channels followed by Batch Normalization layers
	1 Pooling layer to reduce the size by 2

	1 Pooling layer to reduce the size by 8

	3 Fully Connected Layers followed each followed by a dropout layer
'''	
class Model(nn.Module):

	def __init__(self, n_classes, pool_size = 2, n_chars = 70):
		
		super(Model, self).__init__()
		layers, fc_layers = [], []
		n_conv_block_64, n_conv_block_128, n_conv_block_256, n_conv_block_512 = 10, 10, 4, 4
		
		layers.append(nn.Conv1d(n_chars, 64, kernel_size=3, padding=1))

		for i in range(n_conv_block_64):
			layers.append(nn.Conv1d(64, 64, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm1d(64))
			layers.append(nn.ReLU())
			layers.append(nn.Conv1d(64, 64, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm1d(64))
			layers.append(nn.ReLU())

		layers.append(nn.MaxPool1d(pool_size))

		layers.append(nn.Conv1d(64, 128, kernel_size=1))
		layers.append(nn.BatchNorm1d(128))

		#Check this!

		for i in range(n_conv_block_128-1):
			layers.append(nn.Conv1d(128, 128, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm1d(128))
			layers.append(nn.ReLU())
			layers.append(nn.Conv1d(128, 128, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm1d(128))
			layers.append(nn.ReLU())

		layers.append(nn.MaxPool1d(pool_size))

		layers.append(nn.Conv1d(128, 256, kernel_size=1))
		layers.append(nn.BatchNorm1d(256))

		for i in range(n_conv_block_256-1):
			layers.append(nn.Conv1d(256, 256, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm1d(256))
			layers.append(nn.ReLU())
			layers.append(nn.Conv1d(256, 256, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm1d(256))
			layers.append(nn.ReLU())

		layers.append(nn.MaxPool1d(pool_size))
		layers.append(nn.Conv1d(256, 512, kernel_size=1))
		layers.append(nn.BatchNorm1d(512))

		for i in range(n_conv_block_512-1):
			layers.append(nn.Conv1d(512, 512, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm1d(512))
			layers.append(nn.ReLU())
			layers.append(nn.Conv1d(512, 512, kernel_size=3, padding=1))
			layers.append(nn.BatchNorm1d(512))
			layers.append(nn.ReLU())

		layers.append(nn.MaxPool1d(8))

		
		self.final_layer_len = int((max_len - 96) / 27)

		fc_layers.append(nn.Linear(8192, 2048))
		fc_layers.append(nn.Dropout(p = 0.5))
		fc_layers.append(nn.Linear(2048, 2048))
		fc_layers.append(nn.Dropout(p = 0.5))
		fc_layers.append(nn.Linear(2048, n_classes))

		self.layers = nn.Sequential(*layers)
		self.fc_layers = nn.Sequential(*fc_layers)

		#initialisation
		for module in self.modules():
			if isinstance(module, nn.Conv1d):
				module.weight.data.normal_(0, 0.05)

	def forward(self, x):
		x = self.layers(x)
		x = x.view(x.size(0),-1)
		# print(x.size())
		x = self.fc_layers(x)
		return x	