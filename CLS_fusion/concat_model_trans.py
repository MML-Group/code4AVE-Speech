# coding: utf-8
import math
import numpy as np


import torch
import torch.nn as nn
from torch.autograd import Variable


class TransformerGate(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, every_frame=True):
        super(TransformerGate, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.every_frame = every_frame
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(93, 2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=101, nhead=1, batch_first=True)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        # h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        # Forward propagate RNN
        # out, hn = self.gru(x, h0)
        # w = hn[-1, :, :]
        # w = w.unsqueeze(0)
        # batch, 101, 184
        x = self.transformerEncoder(x)
        # x = torch.mean(x, dim=2)
        # x = self.fc(x)
        return x



def TRANSFORMERConcat(mode, inputDim=2048, hiddenDim=2048, nLayers=2, nClasses=500, frameLen=29, every_frame=True):
    model = TransformerGate(inputDim, hiddenDim, nLayers, nClasses, every_frame)
    print('\n'+mode+' model has been built')
    return model
