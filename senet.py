import torch.nn as nn


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
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, k_l=1):
        super(SqueezeExcitationBlockBasic, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.skip_connection = nn.Sequential()
        if in_channels != out_channels:
            # adjust skip connection dimension (dotted lines Lecture 7 slide 37)
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
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


class SqueezeExcitationBlockBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, initial_stride=1, k_l=1):
        super(SqueezeExcitationBlockBottleneck, self).__init__()

        reduced_channels = out_channels // 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, stride=initial_stride, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(reduced_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.se = SqueezeExcitationLayer(out_channels, None, reduction=16)

        self.relu = nn.ReLU(inplace=True)
        self.k_l = k_l

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.se(y)

        y += self.k_l * x
        return self.relu(y)


class SENet(nn.Module):
    def _make_layer(self, block, planes, num_blocks, stride, k_list=None):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        k_sum = 0
        for b in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1, k_l=k_list[b - 1]))
            k_sum += num_blocks
        return nn.Sequential(*layers)

    def __init__(
            self,
            block,
            num_blocks,
            num_classes=10,
            dropout=0.0,
            k_list=None
    ):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.initial_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, k_list=k_list[0:num_blocks[0]])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                       k_list=k_list[num_blocks[0]:sum(num_blocks[0:2])])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                                       k_list=k_list[sum(num_blocks[0:2]):sum(num_blocks[0:3])])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                                       k_list=k_list[sum(num_blocks[0:3]):sum(num_blocks[0:4])])
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.avgp = nn.AvgPool2d(4)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        y = self.initial_block(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avgp(y)
        y = self.dropout(y)
        y = y.view(y.size(0), -1)
        y = self.linear(y)
        return y


def SENet18(dropout=0.0, k_list=None):
    if k_list is None:
        k_list = [1.] * 8
    return SENet(SqueezeExcitationBlockBasic, [2, 2, 2, 2], dropout=dropout, k_list=k_list)


def SENet34(dropout=0.0, k_list=None):
    if k_list is None:
        k_list = [1.] * 16
    return SENet(SqueezeExcitationBlockBasic, [3, 4, 6, 3], dropout=dropout, k_list=k_list)


def SENetBottleneck18(dropout=0.0, k_list=None):
    if k_list is None:
        k_list = [1.] * 8
    return SENet(SqueezeExcitationBlockBottleneck, [2, 2, 2, 2], dropout=dropout, k_list=k_list)


def SENetBottleneck34(dropout=0.0, k_list=None):
    if k_list is None:
        k_list = [1.] * 16
    return SENet(SqueezeExcitationBlockBottleneck, [3, 4, 6, 3], dropout=dropout, k_list=k_list)
