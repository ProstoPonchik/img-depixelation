from torch import nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding="same"):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        skip = self.skip(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += skip

        return self.relu2(x)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ResidualBlock(2, 32)
        self.layer2 = ResidualBlock(32, 64)
        self.layer3 = ResidualBlock(64, 128)
        self.layer4 = ResidualBlock(128, 256)
        self.layer5 = ResidualBlock(256, 512)
        self.layer6 = ResidualBlock(512, 256)
        self.layer7 = ResidualBlock(256, 128)
        self.layer8 = ResidualBlock(128, 64)
        self.layer9 = ResidualBlock(64, 32)
        self.layer10 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        return torch.sigmoid(x)
