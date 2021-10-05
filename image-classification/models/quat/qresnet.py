'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..hypercomplex_layers import QuaternionConv, QuaternionLinear


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = QuaternionConv(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuaternionConv(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QuaternionConv(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = QuaternionConv(in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuaternionConv(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = QuaternionConv(planes, self.expansion * planes, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QuaternionConv(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class QResNet(nn.Module):
    def __init__(self, block, num_blocks, channels=4, num_classes=10):
        super(QResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = QuaternionConv(channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class QResNetLarge(nn.Module):
    def __init__(self, block, num_blocks, channels=4, num_classes=10):
        super(QResNetLarge, self).__init__()
        self.in_planes = 60

        self.conv1 = QuaternionConv(channels, 60, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(60)
        self.layer1 = self._make_layer(block, 60, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 120, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 240, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 516, num_blocks[3], stride=2)
        self.linear = nn.Linear(516*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def QResNet18(channels=4):
    return QResNet(BasicBlock, [2, 2, 2, 2], channels=channels)

def QResNet18Large(channels=4, num_classes=10):
    return QResNetLarge(BasicBlock, [2, 2, 2, 2], channels=channels, num_classes=num_classes)


def QResNet34():
    return QResNet(BasicBlock, [3, 4, 6, 3])


def QResNet50(channels=4):
    return QResNet(Bottleneck, [3, 4, 6, 3], channels=channels)

def QResNet50Large(channels=4, num_classes=10):
    return QResNetLarge(Bottleneck, [3, 4, 6, 3], channels=channels, num_classes=num_classes)


def QResNet101():
    return QResNet(Bottleneck, [3, 4, 23, 3])


def QResNet152():
    return QResNet(Bottleneck, [3, 8, 36, 3])

def QResNet152Large(channels=4, num_classes=10):
    return QResNetLarge(BasicBlock, [3, 8, 36, 3], channels=channels, num_classes=num_classes)


def test():
    net = QResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()