import math

import torch.nn as nn
import torch.nn.init as init
from ..hypercomplex_layers import QuaternionConv, QuaternionLinear

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class QVGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, large=False, num_classes=10):
        super(QVGG, self).__init__()
        self.features = features
        if large:
            filters = [648, 516]
        if not large:
            filters = [512, 512]

        self.classifier = nn.Sequential(
            nn.Dropout(),
            QuaternionLinear(filters[0], filters[1]),
            nn.ReLU(True),
            nn.Dropout(),
            QuaternionLinear(filters[1], filters[1]),
            nn.ReLU(True),
            nn.Linear(filters[1], num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 4
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = QuaternionConv(in_channels, v, kernel_size=3, padding=1, stride=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'A_large': [24, 'M', 72, 'M', 216, 216, 'M', 648, 648, 'M', 648, 648, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
    'E_large': [24, 24, 'M', 72, 72, 'M', 216, 216, 216, 216, 'M', 648, 648, 648, 648, 'M', 
          648, 648, 648, 648, 'M'],
}


def qvgg11():
    """VGG 11-layer model (configuration "A")"""
    return QVGG(make_layers(cfg['A']))


def qvgg11_bn(channels=4):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return QVGG(make_layers(cfg['A'], batch_norm=True))

def qvgg11large_bn(channels=4):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return QVGG(make_layers(cfg['A_large'], batch_norm=True))


def qvgg13():
    """VGG 13-layer model (configuration "B")"""
    return QVGG(make_layers(cfg['B']))


def qvgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return QVGG(make_layers(cfg['B'], batch_norm=True))


def qvgg16():
    """VGG 16-layer model (configuration "D")"""
    return QVGG(make_layers(cfg['D']))


def qvgg16_bn(channels=4):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return QVGG(make_layers(cfg['D'], batch_norm=True))


def qvgg19():
    """VGG 19-layer model (configuration "E")"""
    return QVGG(make_layers(cfg['E']))


def qvgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return QVGG(make_layers(cfg['E'], batch_norm=True))

def qvgg19large_bn(channels=3, num_classes=10):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return QVGG(make_layers(cfg['E_large'], batch_norm=True), large=True, num_classes=num_classes)