import os
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import transforms

"""
=============================自己的包===========================
"""
from BraTS2021 import *
from BraTS2018 import *
from BraTS2019 import *
from utils import *

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


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(args.seed)  # 为所有的GPU设置种子，以使得结果是确定的

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data info
    data_path = "../dataset/brats2021/data"
    test_txt = "../dataset/brats2021/train.txt"
    patch_size = (128, 128, 64)
    test_set = BraTS2021(data_path, test_txt, transform=transforms.Compose([
        # RandomRotFlip(),
        CenterCrop(patch_size),
        GaussianNoise(p=0.1),
        ToTensor()
    ]))

    # data_path = "../dataset/brats2018/data"
    # patch_size = (128, 128, 64)
    # test_set = BraTS2018(data_path, transform=transforms.Compose([
    #     RandomRotFlip(),
    #     CenterCrop(patch_size),
    #     GaussianNoise(p=0.1),
    #     ToTensor()
    # ]))

    print("using {} device.".format(device))

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
    # (86,45) (36,30) (28,45) 45,40 -- train
    d1 = test_set[86]
    slice_w = 45
    image, label = d1
    print("image.shape:", image.shape)
    print("label.shape:", label.shape)

    strat = time.time()
    output = model(image.unsqueeze(dim=0).to(device))
    print(f"one sample cost {time.time()-strat}")
    pre = torch.argmax(output, dim=1)
    print("torch.unique(pre):", torch.unique(pre))
    pre = pre.squeeze(dim=0).cpu().numpy()
    print("pre.shape:", pre.shape)

    # oh_label = F.one_hot(label, 4).permute(3, 0, 1, 2).unsqueeze(0).detach().cpu().numpy()
    # oh_output = torch.argmax(output, dim=1).long()
    # oh_output = F.one_hot(oh_output, 4).permute(0, 4, 1, 2, 3).detach().cpu().numpy()
    # oh_output_h = torch.sigmoid(output).detach().cpu()

    # sen_WT = sensitivity_WT(oh_output, oh_label)
    # sen_ET = sensitivity_ET(oh_output, oh_label)
    # sen_TC = sensitivity_TC(oh_output, oh_label)
    # spe_WT = specificity_WT(oh_output, oh_label)
    # spe_ET = specificity_ET(oh_output, oh_label)
    # spe_TC = specificity_TC(oh_output, oh_label)
    # print("sensitivity_WT:", sen_WT)
    # print("sensitivity_ET:", sen_ET)
    # print("sensitivity_TC:", sen_TC)
    # print("specificity_WT:", spe_WT)
    # print("specificity_ET:", spe_ET)
    # print("specificity_TC:", spe_TC)

    # dis = hausdorff_distance_WT(torch.from_numpy(oh_output), torch.from_numpy(oh_label))
    # print("distance_WT:", dis)
    # dis = hausdorff_dist2ance_ET(torch.from_numpy(oh_output),  torch.from_numpy(oh_label))
    # print("distance_ET:", dis)
    # dis = hausdorff_distance_TC(torch.from_numpy(oh_output),  torch.from_numpy(oh_label))
    # print("distance_TC:", dis)


    # fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(20, 10))
    # ax1.imshow(image[0, :, :, slice_w], cmap='gray')
    # ax1.set_title('Image flair')
    # ax1.axis("off")
    #
    # ax2.imshow(image[1, :, :, slice_w], cmap='gray')
    # ax2.set_title('Image t1ce')
    # ax2.axis("off")
    #
    # ax3.imshow(image[2, :, :, slice_w], cmap='gray')
    # ax3.set_title('Image t1')
    # ax3.axis("off")
    #
    # ax4.imshow(image[3, :, :, slice_w], cmap='gray')
    # ax4.set_title('Image t2')
    # ax4.axis("off")
    #
    # mask_segmentation = label[:, :, slice_w]
    # color_segmentation = np.zeros((patch_size[0], patch_size[0], 3))
    # color_segmentation[mask_segmentation == 1] = [255, 0, 0]  # Red (necrotic tumor core)
    # color_segmentation[mask_segmentation == 2] = [23, 102, 17]  # Green (peritumoral edematous/invaded tissue)
    # color_segmentation[mask_segmentation == 3] = [250, 246, 45]  # Yellow (enhancing tumor)
    # ax5.imshow(color_segmentation.astype('uint8'))
    # # ax5.imshow(image[1, :, :, slice_w], cmap='gray', alpha=0.7)
    # ax5.set_title('Mask')
    # ax5.axis("off")
    #
    # mask_segmentation = pre[:, :, slice_w]
    # color_segmentation = np.zeros((patch_size[0], patch_size[0], 3))
    # color_segmentation[mask_segmentation == 1] = [255, 0, 0]  # Red (necrotic tumor core)
    # color_segmentation[mask_segmentation == 2] = [23, 102, 17]  # Green (peritumoral edematous/invaded tissue)
    # color_segmentation[mask_segmentation == 3] = [250, 246, 45]  # Yellow (enhancing tumor)
    # ax6.imshow(color_segmentation.astype('uint8'))
    # # ax6.imshow(image[1, :, :, slice_w], cmap='gray', alpha=0.7)
    # # ax6.imshow(pre[:, :, slice_w], cmap='gray')
    # ax6.set_title(f'{MODEL_NAME} Prediction')
    # ax6.axis("off")

    f, axarr = plt.subplots(3, 4, figsize=(10, 7))

    # flair
    axarr[0][0].title.set_text('Flair')
    axarr[0][0].imshow(image[0, :, :, slice_w], cmap="gray")
    axarr[0][0].axis('off')

    # t1ce
    axarr[0][1].title.set_text('T1ce')
    axarr[0][1].imshow(image[1, :, :, slice_w], cmap="gray")
    axarr[0][1].axis('off')

    # t1
    axarr[0][2].title.set_text('T1')
    axarr[0][2].imshow(image[2, :, :, slice_w], cmap="gray")
    axarr[0][2].axis('off')

    # t2
    axarr[0][3].title.set_text('T2')
    axarr[0][3].imshow(image[3, :, :, slice_w], cmap="gray")
    axarr[0][3].axis('off')

    # GT
    mask_segmentation = label[:, :, slice_w]
    color_segmentation = np.zeros((patch_size[0], patch_size[1], 3))
    color_segmentation[mask_segmentation == 1] = [255, 0, 0]
    color_segmentation[mask_segmentation == 2] = [23, 102, 17]
    color_segmentation[mask_segmentation == 3] = [250, 246, 45]
    axarr[1][0].imshow(color_segmentation.astype('uint8'))
    # axarr[0][1].imshow(color_segmentation.astype('uint8'), alpha=0.4)
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

    plt.show()


if __name__ == "__main__":
    # MODEL_NAME = 'UNet'
    # MODEL_NAME = 'MyNet'
    # MODEL_NAME = 'Yi_Ding'
    # MODEL_NAME = 'Zhengrong_Luo'
    # MODEL_NAME = 'LiuLiangLiang'
    # MODEL_NAME = 'PeirisHimashi'
    # MODEL_NAME = 'TheophrasteHenry'
    # MODEL_NAME = 'ChenChen'
    MODEL_NAME = 'lslam'
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=20)
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



