 # coding: utf-8
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class Lipreading(nn.Module):
    def __init__(self, hiddenDim=512, embedSize=256):
        super(Lipreading, self).__init__()
        self.inputDim = 512
        self.hiddenDim = hiddenDim
        self.embedSize = embedSize
        self.nLayers = 3
        self.frontend3D = nn.ModuleList([nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),nn.BatchNorm3d(64),nn.ReLU(True),nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))])

        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
        self._initialize_weights()
    
    def forward(self, x):

        frameLen = x.size(2)

        x = self.frontend3D[0](x)
        x = self.frontend3D[1](x)
        x = self.frontend3D[2](x)
        x = self.frontend3D[3](x)

        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet18(x) 

        x = F.dropout(x, p=0.5)        
        x = x.view(-1, frameLen, self.inputDim)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Embedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(Embedding, self).__init__()
        self.fc = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k


        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_k)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))

        self.attention = ScaledDotProductAttention(d_k)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_k, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual, batch_size = x, x.size(0)

        q = self.w_qs(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_ks(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_vs(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)


        output, attn = self.attention(q, k, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.d_k)
        output = self.fc(output)


        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner)

    def forward(self, x):
        enc_output, enc_slf_attn = self.slf_attn(x)
        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn

class Encoder(nn.Module): # to be Encoder !!!
    def __init__(self, d_obs, d_model, d_class, d_k, d_inner, n_head, n_layers):
        super(Encoder, self).__init__()
        self.embedding  = Embedding(input_dim=d_obs, embed_dim=d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k)
            for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_class)
        self.lipreading = Lipreading(hiddenDim=512, embedSize=256)

        # frontend3D
        self.frontend3D = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
                )
        
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(0.5),
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(0.5)
            # nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            # nn.ReLU(True),
            # nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # nn.Dropout3d(0.5)
        )

        # # backend_transformer
        # self.pos_embedding = nn.Parameter(torch.randn(1, 60, 512))
        # self.seq_to_embedding = nn.Linear(2400, 512)
        # self.dropout = nn.Dropout(0.3)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4, batch_first=True, dim_feedforward=2048)
        # self.transformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        # # self.fc = nn.Sequential(
        #     nn.LayerNorm(512),
        #     nn.Linear(512, 101)
        # )


        # initialize
        self.initialize_weights()


    def forward(self, x):
        batch_size = x.size(0)
        # x = self.frontend3D(x) # B, C, T, H, W
        # x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.lipreading(x) # B, T, Dim=512
        # x = x.view(x.size(0), x.size(1), -1) # B, T, Dim=7744
        # x = self.embedding(x)
        
        # model revised!!!
        attentions = []
        for layer in self.layers:
            x, attention = layer(x)
            attentions.append(attention)
        x = x.mean(1)
        x = self.fc(x)
        return x

    def initialize_weights(m):
        for m in m.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        

def lipreading(d_obs, d_model, d_class, d_k, d_inner, n_head, n_layers):
    model = Encoder(d_obs, d_model, d_class, d_k, d_inner, n_head, n_layers)
    return model