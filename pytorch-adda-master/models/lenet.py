"""LeNet model for ADDA."""

import torch.nn.functional as F
from torch import nn
import torch
from torchvision.models.resnet import conv3x3
import math


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        residual = x
        residual = self.bn1(residual)
        residual = self.relu1(residual)
        residual = self.conv1(residual)

        residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)

        if self.downsample is not None:
            x = self.downsample(x)
        return x + residual


class Downsample(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(Downsample, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        assert nOut % nIn == 0
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)


class ResNetEncoder(nn.Module):
    def __init__(self, depth, width=1, block=BasicBlock, classes=10, channels=3):
        self.restored = False
        assert (depth - 2) % 6 == 0
        self.N = (depth - 2) // 6
        super(ResNetEncoder, self).__init__()

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.inplanes = 16
        self.layer1 = self._make_layer(block, 16 * width)
        self.layer2 = self._make_layer(block, 32 * width, stride=2)
        self.layer3 = self._make_layer(block, 64 * width, stride=2)
        self.bn = nn.BatchNorm2d(64 * width)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * width, 500)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Downsample(self.inplanes, planes, stride)
        layers = [block(self.inplanes, planes, stride, downsample=downsample)]
        self.inplanes = planes
        for i in range(self.N - 1):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class LeNetEncoder(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetEncoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [1 x 28 x 28]
            # output [20 x 12 x 12]
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(50 * 4 * 4, 500)

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
        return feat


class LeNetClassifier(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 125),
            nn.ReLU(),
            nn.Linear(125, 9),
        )

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.layer(out)
        return out
