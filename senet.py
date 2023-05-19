import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitationLayer(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(SqueezeExcitationLayer, self).__init__()

        out_channels = in_channels // reduction

        self.squeeze = nn.AdaptiveAvgPool2d(1)

        self.excitation = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.shape
        y = self.squeeze(x).view(batch_size, channels)
        y = self.excitation(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)


class SqueezeExcitationBlockBasic(nn.Module):
    def __init__(self, in_channels, out_channels, initial_stride=1, k_l=1):
        super(SqueezeExcitationBlockBasic, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=initial_stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.skip_connection = nn.Sequential()
        if in_channels != out_channels:
            # adjust skip connection dimension (dotted lines Lecture 7 slide 37)
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=initial_stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()
        self.se = SqueezeExcitationLayer(out_channels, None, reduction=16)
        self.k_l = k_l

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.se(y)
        y += self.k_l * self.skip_connection(x)
        return self.relu(y)
    
    
class SqueezeExcitationBlockFullBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, initial_stride=1, k_l=1):
        super(SqueezeExcitationBlockFullBottleneck, self).__init__()

        reduced_channels = out_channels // 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, stride=initial_stride, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(reduced_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.se = SqueezeExcitationBlock(out_channels, None, reduction=16)

        self.relu = nn.ReLU()
        self.k_l = k_l

    def forward(self, x):
    
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.se(y)

        y += self.k_l * x
        return self.relu(y)

class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)
        self.avgp = nn.AvgPool2d(4)
        self.relu = nn.ReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avgp(y)
        y = y.view(y.size(0), -1)
        y = self.linear(y)
        return y


def SENet18():
    return SENet(SqueezeExcitationBlockBasic, [2,2,2,2])


def SENet34():
    return SENet(SqueezeExcitationBlockBasic, [3,4,6,3])