import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, precision_score, log_loss
from sklearn.model_selection import train_test_split
import re
import sys
import json
import numpy as np

# Usage: python pairwise_ffnn_pytorch.py hidden_ratio emb_dim num_layers training_set

class VerbNet(nn.Module):
    def __init__(self, vocab_size, hidden_ratio=0.5, emb_size=200, num_layers=1):
        super(VerbNet, self).__init__()
        self.emb_size = emb_size
        self.emb_layer = nn.Embedding(vocab_size, self.emb_size)
        self.fc1 = nn.Linear(self.emb_size*2, int(self.emb_size*2*hidden_ratio))
        self.num_layers = num_layers
        if num_layers == 1:
            self.fc2 = nn.Linear(int(self.emb_size*2*hidden_ratio), 1)
        else:
            self.fc2 = nn.Linear(int(self.emb_size*2*hidden_ratio), int(self.emb_size*hidden_ratio))
            self.fc3 = nn.Linear(int(self.emb_size*hidden_ratio), 1)
        self.is_training = True
        
    def forward(self, x):
        x_emb = self.emb_layer(x)
        fullX = torch.cat((x_emb[:,0,:], x_emb[:,1,:]), dim=1)
        layer1 = F.relu(self.fc1(F.dropout(fullX, p=0.3, training=self.is_training)))
        if self.num_layers == 1:
            return torch.sigmoid(self.fc2(layer1))
        layer2 = F.relu(self.fc2(F.dropout(layer1, p=0.3, training=self.is_training)))
        layer3 = torch.sigmoid(self.fc3(layer2))
        return layer3

    def retrieveEmbeddings(self,x):
        x_emb = self.emb_layer(x)
        fullX = torch.cat((x_emb[:, 0, :], x_emb[:, 1, :]), dim=1)
        layer1 = F.relu(self.fc1(fullX))
        if self.num_layers == 1:
            return layer1
        layer2 = F.relu(self.fc2(layer1))

        return torch.cat((layer1,layer2),1)
