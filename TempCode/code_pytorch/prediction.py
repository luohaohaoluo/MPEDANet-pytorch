import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


"""
=============================自己的包===========================
"""
from BraTS2021 import *
from utils import *

from networks.Unet import UNet
from networks.MyNet.MyNet import MyNet
from networks.DingYi.Ding_Yi import Yi_Ding
from networks.Liuliangliang.Liuliangliang import Liangliang_Liu
from networks.LuoZhengrong.LuoZhengrong import Zhengrong_Luo
from networks.PeirisHimashi.unet import Unet
from networks.PeirisHimashi.layers import get_norm_layer
from networks.ChenChen.DMFNet import DMFNet
from networks.Islam.attention_unet import UNet3D
from networks.Henry.unet import Att_EquiUnet


def load_pic(path, trans=None):
    label = sitk.GetArrayFromImage(
        sitk.ReadImage(path + '/' + f'{path[-15:]}_seg.nii.gz')).transpose(1, 2, 0)
    # print(label.shape)
    # 堆叠四种模态的图像，4 x (H,W,D) -> (4,H,W,D)
    images = np.stack(
        [sitk.GetArrayFromImage(
            sitk.ReadImage(path + '/' + f'{path[-15:]}{modal}.nii.gz')).transpose(1, 2, 0) for
         modal in
         modalities],
        0)  # [240,240,155]

    # 数据类型转换
    label = label.astype(np.uint8)
    images = images.astype(np.float32)

    # 对第一个通道求和，如果四个模态都为0，则标记为背景(False)
    mask = images.sum(0) > 0
    for k in range(4):
        x = images[k, ...]  #
        y = x[mask]

        # 对背景外的区域进行归一化
        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[k, ...] = x

    # [0,1,2,4] -> [0,1,2,3]
    label[label == 4] = 3
    sample = {'image': images, 'label': label}

    if trans:
        sample = trans(sample)
    return sample['image'], sample['label']


def load_pic_without_nomal(path, trans=None):
    label = sitk.GetArrayFromImage(
        sitk.ReadImage(path + '/' + f'{path[-15:]}_seg.nii.gz')).transpose(1, 2, 0)
    # print(label.shape)
    # 堆叠四种模态的图像，4 x (H,W,D) -> (4,H,W,D)
    images = np.stack(
        [sitk.GetArrayFromImage(
            sitk.ReadImage(path + '/' + f'{path[-15:]}{modal}.nii.gz')).transpose(1, 2, 0) for
         modal in
         modalities],
        0)  # [240,240,155]

    # 数据类型转换
    label = label.astype(np.uint8)
    images = images.astype(np.float32)

    # [0,1,2,4] -> [0,1,2,3]
    label[label == 4] = 3
    sample = {'image': images, 'label': label}

    if trans:
        sample = trans(sample)
    return sample['image'], sample['label']


def main(args, saving=False):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(args.seed)  # 为所有的GPU设置种子，以使得结果是确定的

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))

    # data info
    case = '00084'
    data_path = f"../dataset/brats2021/data/BraTS2021_{case}"
    patch_size = (160, 160, 64)
    image, label = load_pic(data_path, trans=transforms.Compose([
        CenterCrop(patch_size),
        ToTensor()
    ]))
    real_image, _ = load_pic_without_nomal(data_path, transforms.Compose([
        CenterCrop(patch_size),
        ToTensor()
    ]))

    print("image.shape:", image.shape)
    print("label.shape:", label.shape)
    print(np.unique(label))

    # 1-坏疽(NT,necrotic tumor core),2-浮肿区域(ED,peritumoral edema),4-增强肿瘤区域(ET,enhancing tumor)
    # 评价指标：ET(label4),TC(label1+label4),WT(label1+label2+label4)

    print(f"using {MODEL_NAME} for prediction")
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
    # 加载训练模型
    if os.path.exists(args.weights):
        weight_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(weight_dict['model'])
        print('Successfully loading checkpoint.')

    model.eval()

    output = model(image.unsqueeze(dim=0).to(device))
    print(output.shape)
    pre = torch.argmax(output, dim=1)
    print(torch.unique(pre))
    pre = pre.squeeze(dim=0).cpu().numpy()
    print(pre.shape)

    slice_w = 40

    f, axarr = plt.subplots(3, 4, figsize=(10, 7))

    # flair
    axarr[0][0].title.set_text('Flair')
    axarr[0][0].imshow(real_image[0, :, :, slice_w], cmap="gray")
    axarr[0][0].axis('off')

    # t1ce
    axarr[0][1].title.set_text('T1ce')
    axarr[0][1].imshow(real_image[1, :, :, slice_w], cmap="gray")
    axarr[0][1].axis('off')

    # t1
    axarr[0][2].title.set_text('T1')
    axarr[0][2].imshow(real_image[2, :, :, slice_w], cmap="gray")
    axarr[0][2].axis('off')

    # t2
    axarr[0][3].title.set_text('T2')
    axarr[0][3].imshow(real_image[3, :, :, slice_w], cmap="gray")
    axarr[0][3].axis('off')

    # GT
    mask_segmentation = label[:, :, slice_w]
    color_segmentation = np.zeros((patch_size[0], patch_size[1], 3))
    color_segmentation[mask_segmentation == 1] = [255, 0, 0]
    color_segmentation[mask_segmentation == 2] = [23, 102, 17]
    color_segmentation[mask_segmentation == 3] = [250, 246, 45]
    axarr[1][0].imshow(color_segmentation.astype('uint8'))
    axarr[0][1].imshow(color_segmentation.astype('uint8'), alpha=0.4)
    axarr[1][0].title.set_text('Ground truth')
    axarr[1][0].axis('off')

    # pre all classes
    mask_segmentation = pre[:, :, slice_w]
    color_segmentation = np.zeros((patch_size[0], patch_size[1], 3))
    color_segmentation[mask_segmentation == 1] = [255, 0, 0]
    color_segmentation[mask_segmentation == 2] = [23, 102, 17]
    color_segmentation[mask_segmentation == 3] = [250, 246, 45]
    axarr[1][1].imshow(color_segmentation.astype('uint8'), cmap="gray")
    axarr[1][1].title.set_text('pre all classes')
    axarr[1][1].axis('off')

    # WT 1,2,4
    mask_segmentation = label[:, :, slice_w]
    color_segmentation = np.zeros((patch_size[0], patch_size[1], 3))
    color_segmentation[mask_segmentation == 1] = [255, 255, 255]
    color_segmentation[mask_segmentation == 2] = [255, 255, 255]
    color_segmentation[mask_segmentation == 3] = [255, 255, 255]
    axarr[1][2].imshow(color_segmentation.astype('uint8'), cmap="gray")
    axarr[1][2].title.set_text('GT WT')
    axarr[1][2].axis('off')

    # pre WT
    mask_segmentation = pre[:, :, slice_w]
    color_segmentation = np.zeros((patch_size[0], patch_size[1], 3))
    color_segmentation[mask_segmentation == 1] = [255, 255, 255]
    color_segmentation[mask_segmentation == 2] = [255, 255, 255]
    color_segmentation[mask_segmentation == 3] = [255, 255, 255]
    axarr[1][3].imshow(color_segmentation.astype('uint8'), cmap="gray")
    axarr[1][3].title.set_text('pre WT')
    axarr[1][3].axis('off')

    # GT TC 1,4
    mask_segmentation = label[:, :, slice_w]
    color_segmentation = np.zeros((patch_size[0], patch_size[1], 3))
    color_segmentation[mask_segmentation == 1] = [255, 255, 255]
    color_segmentation[mask_segmentation == 3] = [255, 255, 255]
    axarr[2][0].imshow(color_segmentation.astype('uint8'), cmap="gray")
    axarr[2][0].title.set_text('GT TC')
    axarr[2][0].axis('off')

    # pre TC
    mask_segmentation = pre[:, :, slice_w]
    color_segmentation = np.zeros((patch_size[0], patch_size[1], 3))
    color_segmentation[mask_segmentation == 1] = [255, 255, 255]
    color_segmentation[mask_segmentation == 3] = [255, 255, 255]
    axarr[2][1].imshow(color_segmentation.astype('uint8'), cmap="gray")
    axarr[2][1].title.set_text('pre TC')
    axarr[2][1].axis('off')

    # GT ET 4
    mask_segmentation = label[:, :, slice_w]
    color_segmentation = np.zeros((patch_size[0], patch_size[1], 3))
    color_segmentation[mask_segmentation == 3] = [255, 255, 255]
    axarr[2][2].imshow(color_segmentation.astype('uint8'), cmap="gray")
    axarr[2][2].title.set_text('GT ET')
    axarr[2][2].axis('off')

    # pre ET
    mask_segmentation = pre[:, :, slice_w]
    color_segmentation = np.zeros((patch_size[0], patch_size[1], 3))
    color_segmentation[mask_segmentation == 3] = [255, 255, 255]
    axarr[2][3].imshow(color_segmentation.astype('uint8'), cmap="gray")
    axarr[2][3].title.set_text('pre ET')
    axarr[2][3].axis('off')

    if saving:
        plt.savefig("./result/UNet")

    plt.show()


if __name__ == "__main__":
    # MODEL_NAME = 'UNet'
    MODEL_NAME = 'MyNet'
    # MODEL_NAME = 'PeirisHimashi'
    # MODEL_NAME = 'Yi_Ding'
    # MODEL_NAME = 'Zhengrong_Luo'
    # MODEL_NAME = 'LiuLiangLiang'
    # MODEL_NAME = 'TheophrasteHenry'
    # MODEL_NAME = 'ChenChen'
    # MODEL_NAME = 'lslam'
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=0.002)
    parser.add_argument('--data_path', type=str, default='../dataset/brats2021/data')
    # parser.add_argument('--val_data_path', type=str, default='../dataset/brats2020/val_data/data')
    parser.add_argument('--train_txt', type=str, default='../dataset/brats2021/train.txt')
    parser.add_argument('--valid_txt', type=str, default='../dataset/brats2021/valid.txt')
    # parser.add_argument('--test_txt', type=str, default='../dataset/brats2021/test.txt')
    parser.add_argument('--train_log', type=str, default=f'results/{MODEL_NAME}/{MODEL_NAME}.txt')
    parser.add_argument('--weights', type=str, default=f'results/{MODEL_NAME}/{MODEL_NAME}.pth')
    parser.add_argument('--save_path', type=str, default=f'checkpoint/{MODEL_NAME}')

    args = parser.parse_args()

    main(args)



