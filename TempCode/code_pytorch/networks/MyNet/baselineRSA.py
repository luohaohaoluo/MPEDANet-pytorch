import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class RSA_Block(nn.Module):
    def __init__(self, ou_ch):
        super(RSA_Block, self).__init__()
        self.scaler = 8

        self.conv_k = nn.Conv3d(ou_ch, ou_ch//self.scaler, kernel_size=1)
        self.conv_q = nn.Conv3d(ou_ch, ou_ch//self.scaler, kernel_size=1)

        self.conv_v = nn.Conv3d(ou_ch, ou_ch, kernel_size=1)
        self.identity = nn.Identity()
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = self.identity(x)

        x_k = self.conv_k(x)
        x_q = self.conv_q(x)
        x_v = self.conv_v(x)

        x_k = rearrange(x_k, 'b c d h w -> b (c d) (h w)')
        x_q = rearrange(x_q, 'b c d h w -> b (c d) (h w)')
        x_v = rearrange(x_v, 'b c d h w -> b (c d) (h w)')
        # print("k:", x_k.shape)
        # print("v:", x_v.shape)

        x_k = x_k.transpose(1, 2)
        # [1, 16384, 256], [1, 256, 16384] -> [1, 16384, 16384]
        dot = torch.einsum('b i j , b j d -> b i d', x_k, x_q)
        attention = F.softmax(dot, dim=-1).transpose(1, 2)
        # print(attention.shape)

        # [1, 2048, 16384], [1, 16384, 16384] -> [1, 2048, 16384]
        x_a = torch.einsum('b i j, b j d-> b i d', x_v, attention)
        # print(x_a.shape)
        x_a = rearrange(x_a, 'b (c d) (h w) -> b c d h w', d=x.shape[-3], h=x.shape[-2])
        # print(x_a.shape)
        # print(identity.shape)
        identity = self.gamma * identity + x_a

        return identity


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_ch, out_ch)
        )
        # self.rsa = RSA_Block(out_ch)

    def forward(self, x):
        x = self.mpconv(x)
        # x = self.rsa(x)
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


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
        self.rsa = RSA_Block(in_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        x = self.rsa(x)
        x = self.conv(x)

        return x


class BaselineRSA(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(BaselineRSA, self).__init__()
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
    net = BaselineRSA(in_channels=4, num_classes=4)
    y = net(x)
    print("params: ", sum(p.numel() for p in net.parameters()))
    print(y.shape)















