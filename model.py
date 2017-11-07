import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Model for the character-based CNN
class Model(nn.Module):

	def __init__(self, n_classes, pool_size = 3, n_chars = 70, channel_size = 256, kernel = [7, 7, 3, 3, 3, 3], max_len = 1014):
		
		super(Model, self).__init__()
	
		self.conv1 = nn.Conv1d(n_chars, channel_size, kernel[0])
		self.pool1 = nn.MaxPool1d(pool_size)
		self.conv2 = nn.Conv1d(channel_size, channel_size, kernel[1])
		self.pool2 = nn.MaxPool1d(pool_size)
		self.conv3 = nn.Conv1d(channel_size, channel_size, kernel[2])
		self.conv5 = nn.Conv1d(channel_size, channel_size, kernel[3])
		self.conv4 = nn.Conv1d(channel_size, channel_size, kernel[4])
		self.conv6 = nn.Conv1d(channel_size, channel_size, kernel[5])
		self.pool3 = nn.MaxPool1d(pool_size)
		
		self.final_layer_len = int((max_len - 96) / 27)
		self.FC1 = nn.Linear(channel_size * self.final_layer_len, 1024)
		self.FC2 = nn.Linear(1024, 1024)
		self.final_layer = nn.Linear(1024, n_classes)
		self.dropout1 = nn.Dropout(p = 0.5)
		self.dropout2 = nn.Dropout(p = 0.5)

		#initialisation
		for module in self.modules():
			if isinstance(module, nn.Conv1d):
				module.weight.data.normal_(0, 0.05)

	def forward(self, x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.conv6(x)
		x = self.pool3(x)
		x = x.view(x.size(0),-1)		
		x = self.FC1(x)
		x = self.dropout1(x)
		x = self.FC2(x)
		x = self.dropout2(x)
		x = self.final_layer(x)
		x = F.softmax(x)
		return x		