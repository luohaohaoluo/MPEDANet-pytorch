import torch
import numpy as np

from thop import profile
from networks.Unet import UNet
from networks.MyNet.MyNet import MyNet
from networks.DingYi.Ding_Yi import Yi_Ding
from networks.Liuliangliang.Liuliangliang import Liangliang_Liu
from networks.LuoZhengrong.LuoZhengrong import Zhengrong_Luo
from networks.PeirisHimashi.unet import Unet
from networks.PeirisHimashi.layers import get_norm_layer
from networks.Henry.unet import Att_EquiUnet
from networks.ChenChen.DMFNet import DMFNet
from networks.Islam.attention_unet import UNet3D


if __name__ == '__main__':
    device = torch.device('cuda')
    # MODEL_NAME = 'MyNet'
    # MODEL_NAME = 'UNet'
    # MODEL_NAME = 'Yi_Ding'
    # MODEL_NAME = 'Zhengrong_Luo'
    # MODEL_NAME = 'LiuLiangLiang'
    # MODEL_NAME = 'PeirisHimashi'
    MODEL_NAME = 'TheophrasteHenry'
    # MODEL_NAME = 'ChenChen'
    # MODEL_NAME = 'lslam'
    if MODEL_NAME == 'MyNet':
        model = MyNet(in_channels=4, num_classes=4).to(device)
    elif MODEL_NAME == 'UNet':
        model = UNet(in_channels=4, num_classes=4).to(device)
    elif MODEL_NAME == 'Yi_Ding':
        model = Yi_Ding(in_data=4, out_data=4).to(device)
    elif MODEL_NAME == 'Zhengrong_Luo':
        model = Zhengrong_Luo(in_data=4, out_data=4).to(device)
    elif MODEL_NAME == 'LiuLiangLiang':
        model = Liangliang_Liu(in_data=4, out_data=4).to(device)
    elif MODEL_NAME == 'PeirisHimashi':
        model = Unet(4, 4, width=32, norm_layer=get_norm_layer('inorm'), dropout=0).to(device)
    elif MODEL_NAME == 'TheophrasteHenry':
        model = Att_EquiUnet(4, 4, width=32, norm_layer=get_norm_layer('inorm'), dropout=0).to(device)
    elif MODEL_NAME == 'ChenChen':
        model = DMFNet(c=4, groups=16, norm='sync_bn', num_classes=4).to(device)
    elif MODEL_NAME == 'lslam':
        model = UNet3D(in_channels=4, out_channels=4, final_sigmoid=False).to(device)

    # print("model params: ", sum(p.numel() for p in model.parameters()))

    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params / 1e6}M')
    print(f'Trainable params: {Trainable_params / 1e6}M')
    print(f'Non-trainable params: {NonTrainable_params / 1e6}M')

    input1 = torch.randn(1, 4, 128, 128, 64).to(device)
    flops, params = profile(model, inputs=(input1,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

