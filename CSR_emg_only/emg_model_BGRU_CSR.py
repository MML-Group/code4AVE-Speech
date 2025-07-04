# coding: utf-8
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


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
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=21, padding=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
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
        # x = self.avgpool(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, x.size(2))
        x = self.fc(x)


        return x


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, every_frame=True):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.every_frame = every_frame

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        out, _ = self.gru(x, h0)
        return out


class EMGNet(nn.Module):
    def __init__(self, mode, inputDim=256, hiddenDim=512, nClasses=101, frameLen=29, every_frame=True):
        super(EMGNet, self).__init__()
        self.mode = mode
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.every_frame = every_frame
        self.nLayers = 2


        self.fronted2D = nn.Sequential(
                nn.Conv2d(6, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
                # nn.MaxPool2d(kernel_size=2, stride=2),
                # nn.Dropout(0.25)
                )
        self.fronted2D1 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.5)
                )
        self.fronted2D2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
                # nn.MaxPool2d(kernel_size=2, stride=2),
                # nn.Dropout(0.25)
                )
        self.fronted2D3 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.5)
                )
        self.fronted2D4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # nn.MaxPool2d(kernel_size=3, stride=3),
                # nn.Dropout(0.5)
                )
        self.fronted2D5 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.5)
                )
        self.fronted2D6 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # nn.MaxPool2d(kernel_size=2, stride=2),
                # nn.Dropout(0.25)
                )
        self.fronted2D7 = nn.Sequential(
                nn.Conv2d(256,256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=3),
                nn.Dropout(0.5)
                )

        self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses, self.every_frame)

        self.fc_csr = nn.Linear(1024, 194 + 1)
        # initialize
        self._initialize_weights()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.fronted2D(x)
        x = self.fronted2D1(x)
        x = self.fronted2D2(x)
        x = self.fronted2D3(x) ## 128,9,9
        x = self.fronted2D4(x) ## 256,9,9
        x = self.fronted2D5(x) ## 256,4,4
        x = x.contiguous()
        # x = self.fronted2D6(x)
        # x = self.fronted2D7(x)

        x = x.view(batch_size, -1, self.inputDim)
        x = self.gru(x)
        x = self.fc_csr(x)
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


def emg_model(mode, inputDim=256, hiddenDim=512, nClasses=101, frameLen=36, every_frame=True):
        model = EMGNet(mode, inputDim=inputDim, hiddenDim=hiddenDim, nClasses=nClasses, frameLen=frameLen, every_frame=every_frame)
        return model


if __name__ == "__main__":
    model = emg_model('finetuneGRU', inputDim=256, hiddenDim=512, nClasses=101, frameLen=36, every_frame=True)
    x = torch.randn((16,6,36,36))
    y = model(x)
    # y = torch.mean(y, 1)
    print(y.size())
