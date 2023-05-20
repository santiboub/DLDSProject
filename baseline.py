import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, dropout=0, increased_dropout=0, batch_norm=False, num_classes=10):
        super(BaselineModel, self).__init__()

        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32) if batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64) if batch_norm else nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
        self.block1.apply(init_weights)

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64) if batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout + increased_dropout)
        )
        self.block2.apply(init_weights)

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout + increased_dropout * 2)
        )
        self.block3.apply(init_weights)

        self.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128) if batch_norm else nn.Identity(),
            nn.Dropout(dropout + increased_dropout * 3),
            nn.Linear(128, num_classes)
        )
        self.fc.apply(init_weights)

        #self.to(device)  # what is this?

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        # x4 = x3.view(-1, 64 * 8 * 8)
        current_batch_size = x3.size(dim=0)
        x4 = x3.reshape((current_batch_size, -1))

        x5 = self.fc(x4)

        return x5


class BaselineModelModifiedBNDropoutOrder(nn.Module):
    def __init__(self, dropout=0, increased_dropout=0, batch_norm=False, num_classes=10):
        super(BaselineModelModifiedBNDropoutOrder, self).__init__()

        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.BatchNorm2d(32) if batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.BatchNorm2d(64) if batch_norm else nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block1.apply(init_weights)

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout((dropout + increased_dropout) / 2),
            nn.BatchNorm2d(64) if batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout((dropout + increased_dropout) / 2),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2.apply(init_weights)

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout((dropout + increased_dropout * 2) / 2),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout((dropout + increased_dropout * 2) / 2),
            nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3.apply(init_weights)

        self.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout + increased_dropout * 3),
            nn.BatchNorm1d(128) if batch_norm else nn.Identity(),
            nn.Linear(128, num_classes)
        )
        self.fc.apply(init_weights)

        #self.to(device)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        current_batch_size = x3.size(dim=0)
        x4 = x3.reshape((current_batch_size, -1))
        x5 = self.fc(x4)

        return x5