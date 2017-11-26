import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable

# Model for the character-based CNN having 6 Convolutional layers, as described in the paper https://arxiv.org/abs/1509.01626
class Model(nn.Module):

	def __init__(self, n_classes, pool_size = 3, n_chars = 70, channel_size = 256, kernel = [7, 7, 3, 3, 3, 3], max_len = 1014):
		
		super(Model, self).__init__()
		layers, fc_layers = [], []

		# The Conv/Pool layers
		layers.append(nn.Conv1d(n_chars, channel_size, kernel[0]))
		layers.append(nn.ReLU())
		layers.append(nn.MaxPool1d(pool_size))
		layers.append(nn.Conv1d(channel_size, channel_size, kernel[1]))
		layers.append(nn.ReLU())
		layers.append(nn.MaxPool1d(pool_size))
		for i in range(2,6):
			layers.append(nn.Conv1d(channel_size, channel_size, kernel[i]))
			layers.append(nn.ReLU())
		layers.append(nn.MaxPool1d(pool_size))
		
		self.final_layer_len = int((max_len - 96) / 27)

		# The Fully Connected layers, with dropout
		fc_layers.append(nn.Linear(channel_size * self.final_layer_len, 1024))
		fc_layers.append(nn.ReLU())
		fc_layers.append(nn.Dropout(p = 0.5))
		fc_layers.append(nn.Linear(1024, 1024))
		fc_layers.append(nn.ReLU())
		fc_layers.append(nn.Dropout(p = 0.5))
		fc_layers.append(nn.Linear(1024, n_classes))

		self.layers = nn.Sequential(*layers)
		self.fc_layers = nn.Sequential(*fc_layers)

		#initialisation
		for module in self.modules():
			if isinstance(module, nn.Conv1d):
				module.weight.data.normal_(0, 0.05)

	def forward(self, x):
		x = self.layers(x)
		x = x.view(x.size(0),-1)	
		x = self.fc_layers(x)
		return x	