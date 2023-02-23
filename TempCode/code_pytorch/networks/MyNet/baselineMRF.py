import torch
import torch.nn as nn


class ERF_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ERF_Block, self).__init__()

        self.step1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(out_ch//2, out_ch)
        )

        self.step2 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(out_ch//2, out_ch)
        )

        self.step3 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.GroupNorm(out_ch//2, out_ch)
        )

        self.step2_3 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(out_ch//2, out_ch)
        )

    def forward(self, x):

        x1 = self.step1(x)
        x2 = self.step2(x)
        x3 = self.step3(x)

        x2_3 = self.step2_3(x2 + x3)

        out = x2_3 + x1
        # out = x1 + x2 + x3

        return out


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = ERF_Block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2, 2),
            ERF_Block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        # x = self.sigmoid(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv1 = ERF_Block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        return x


class BaselineMRF(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(BaselineMRF, self).__init__()
        features = [8, 16, 32, 64]

        self.inc = InConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        self.up1 = Up(features[3], features[2])
        self.up2 = Up(features[2], features[1])
        self.up3 = Up(features[1], features[0])
        self.outc = OutConv(features[0], num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 4, 160, 160, 128)
    net = BaselineMRF(in_channels=4, num_classes=4)
    y = net(x)
    print("params: ", sum(p.numel() for p in net.parameters()))
    print(y.shape)















