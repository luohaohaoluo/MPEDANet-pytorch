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


class ERF_Block(nn.Module):
    "called MRF_Block"
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


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()

        self.conv1 = ERF_Block(in_ch, out_ch)

        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.maxpool(x)

        x = self.conv1(x)

        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, flag):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv1 = ERF_Block(in_ch, out_ch)
        self.rsa = RSA_Block(in_ch)

        if flag == 'a':
            self.skip = nn.Conv3d(out_ch, out_ch//2, kernel_size=1)
        elif flag == 'b':
            self.skip = nn.Conv3d(out_ch, out_ch, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.skip is not None:
            x2 = self.skip(x2)
            x2 = self.relu(x2)

        x = torch.cat([x2, x1], dim=1)

        x = self.rsa(x)
        x = self.conv1(x)

        return x


class MyNet(nn.Module):
    """
    EAMRNet: Enhance attention and multiple Reception Net
    """
    def __init__(self, in_channels, num_classes):
        super(MyNet, self).__init__()
        self.channel_list = [8, 16, 32, 64]

        self.in_conv1 = ERF_Block(in_channels, self.channel_list[0])

        self.down1 = Down(self.channel_list[0], self.channel_list[1])
        self.down2 = Down(self.channel_list[1], self.channel_list[2])
        self.down3 = Down(self.channel_list[2], self.channel_list[3])

        self.up1 = Up(self.channel_list[3], self.channel_list[2], 'b')
        self.up2 = Up(self.channel_list[2], self.channel_list[1], 'b')
        self.up3 = Up(self.channel_list[1], self.channel_list[0], 'b')

        self.out_conv = nn.Conv3d(self.channel_list[0], num_classes, kernel_size=1)

        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.in_conv1(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.out_conv(x)

        return x


if __name__ == '__main__':
    # x = torch.randn(1, 128, 128, 128, 64)
    # net = RSA_Block(128)
    # y = net(x)
    # print("params: ", sum(p.numel() for p in net.parameters()))
    # print(y.shape)

    # x = torch.randn(1, 32, 128, 128, 64)
    # net = ERF_Block(32, 64)
    # y = net(x)
    # print("params: ", sum(p.numel() for p in net.parameters()))
    # print(y.shape)

    x = torch.randn(1, 4, 128, 128, 64)
    net = MyNet(in_channels=4, num_classes=4)
    y = net(x)
    print("params: ", sum(p.numel() for p in net.parameters()))
    print(y.shape)


