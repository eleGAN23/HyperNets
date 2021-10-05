import math

import torch.nn as nn
import torch.nn.init as init
from ..hypercomplex_layers import PHConv, PHMLinear

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class PHCVGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, n, large=False, num_classes=10):
        super(PHCVGG, self).__init__()
        self.features = features
        if large:
            filters = [648, 516]
        if not large:
            filters = [512, 512]
        self.classifier = nn.Sequential(
            nn.Dropout(),
            PHMLinear(n, filters[0], filters[1]),
            nn.ReLU(True),
            nn.Dropout(),
            PHMLinear(n, filters[1], filters[1]),
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


def make_layers(cfg, batch_norm=False, n=1, channels=3):
    layers = []
    in_channels = channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = PHConv(n, in_channels, v, kernel_size=3, padding=1)
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


def phcvgg11():
    """VGG 11-layer model (configuration "A")"""
    return PHCVGG(make_layers(cfg['A']))


def phcvgg11_bn(channels=3, n=1):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return PHCVGG(make_layers(cfg['A'], batch_norm=True, n=n, channels=channels))

def phcvgg11large_bn(channels=3, n=1):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return PHCVGG(make_layers(cfg['A_large'], batch_norm=True, n=n, channels=channels))


def phcvgg13():
    """VGG 13-layer model (configuration "B")"""
    return PHCVGG(make_layers(cfg['B']))


def phcvgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return PHCVGG(make_layers(cfg['B'], batch_norm=True))


def phcvgg16():
    """VGG 16-layer model (configuration "D")"""
    return PHCVGG(make_layers(cfg['D']))


def phcvgg16_bn(channels=4, n=1):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return PHCVGG(make_layers(cfg['D'], batch_norm=True, n=n, channels=channels), n=n)


def phcvgg19():
    """VGG 19-layer model (configuration "E")"""
    return PHCVGG(make_layers(cfg['E']))


def phcvgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return PHCVGG(make_layers(cfg['E'], batch_norm=True))

def phcvgg19large_bn(channels=3, n=1, num_classes=10):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return PHCVGG(make_layers(cfg['E_large'], batch_norm=True, n=n, channels=channels), n=n, large=True, num_classes=num_classes)