# coding: utf-8
import math
import numpy as np


import torch
import torch.nn as nn
from torch.autograd import Variable

class PositionalEncoding(nn.Module):

    """
    A layer to add positional encodings to the inputs of a Transformer model.
    Formula:
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """

    def __init__(self, dModel, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float()*(math.log(10000.0)/dModel))
        pe[:, 0::2] = torch.sin(position/denominator)
        pe[:, 1::2] = torch.cos(position/denominator)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pe", pe)


    def forward(self, inputBatch):
        outputBatch = inputBatch + self.pe[:inputBatch.shape[0],:,:]
        return outputBatch


class TransformerGate(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, every_frame=True):
        super(TransformerGate, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.every_frame = every_frame

        self.fc_expand = nn.Linear(195, 512)

        self.embeddings = nn.Embedding(197, 512) ## 194 + 1 + 2
        self.pos_encode = PositionalEncoding(dModel=512, maxLen=92)
        self.pos_decode = PositionalEncoding(dModel=512, maxLen=6)


        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.FC    = nn.Linear(512, 194+1)
        self.FC_attn = nn.Linear(512, 194+3)

    def forward(self, x, tgt, tgt_mask, tgt_padding_mask):

        x = x.transpose(0,1) # T, B, D
        x = self.fc_expand(x)


        # TM Encoder
        x = self.pos_encode(x)
        encoder_output = self.encoder(x) ## T, B, 512

        #tgt embedding module
        tgt = self.embeddings(tgt) #tgt=(B, T), embedding=(B,T,D)
        tgt = tgt.transpose(0,1) #(T,B,D)
        tgt = self.pos_decode(tgt) #(T,B,D)

        # CTC loss
        y_ctc = self.FC(encoder_output)
        y_ctc = y_ctc.permute(1, 0, 2).contiguous() # B, T, D

        # CE loss
        y_ce = self.decoder(tgt=tgt, memory=encoder_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        y_ce = self.FC_attn(y_ce)
        y_ce = y_ce.transpose(0, 1) # B, T, D

        return y_ctc, y_ce



def TRANSFORMERConcat(mode, inputDim=2048, hiddenDim=2048, nLayers=2, nClasses=500, frameLen=29, every_frame=True):
    model = TransformerGate(inputDim, hiddenDim, nLayers, nClasses, every_frame)
    print('\n'+mode+' model has been built')
    return model
