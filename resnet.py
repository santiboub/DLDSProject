import torch.nn as nn


class ResnetBlockBasic(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, k_l=1):
        super(ResnetBlockBasic, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
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
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()
        self.k_l = k_l

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y += self.k_l * self.skip_connection(x)
        return self.relu(y)


class ResnetBlockBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, initial_stride=1, k_l=1):
        super(ResnetBlockBottleneck, self).__init__()

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

        self.relu = nn.ReLU()
        self.k_l = k_l

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        y += self.k_l * x
        return self.relu(y)


class ResNet(nn.Module):
    def _make_layer(self, block, planes, num_blocks, stride):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.initial_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)
        self.avgp = nn.AvgPool2d(4)

    def forward(self, x):
        y = self.initial_block(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avgp(y)
        y = y.view(y.size(0), -1)
        y = self.linear(y)
        return y


def ResNet18():
    return ResNet(ResnetBlockBasic, [2, 2, 2, 2])


def ResNet34():
    return ResNet(ResnetBlockBasic, [3, 4, 6, 3])
