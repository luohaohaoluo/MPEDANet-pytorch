import os
import argparse

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
from utils import Loss, cal_dice

from networks.MyNet.MyNet import MyNet
from networks.MyNet.baseline import Baseline
from networks.MyNet.baselineMRF import BaselineMRF
from networks.MyNet.baselineRSA import BaselineRSA
from networks.DingYi.Ding_Yi import Yi_Ding
from networks.Liuliangliang.Liuliangliang import Liangliang_Liu
from networks.LuoZhengrong.LuoZhengrong import Zhengrong_Luo
from networks.PeirisHimashi.unet import Unet
from networks.PeirisHimashi.layers import get_norm_layer
from networks.Henry.unet import Att_EquiUnet
from networks.ChenChen.DMFNet import DMFNet
from networks.Islam.attention_unet import UNet3D


def train_loop(model, optimizer, scheduler, criterion, train_loader, device, epoch):
    model.train()
    running_loss = 0
    dice1_train = 0
    dice2_train = 0
    dice3_train = 0
    pbar = tqdm(train_loader, desc='Training: ', colour='#3b96b8', unit='sample')
    for it, (images, masks) in enumerate(pbar):
        # update learning rate according to the schedule
        # it = len(train_loader) * epoch + it
        # param_group = optimizer.param_groups[0]
        # param_group['lr'] = scheduler[it]

        # print(scheduler[it])

        # [b,4,128,128,128] , [b,128,128,128]
        images, masks = images.to(device), masks.to(device)
        # [b,4,128,128,128], 4分割
        outputs = model(images)
        # outputs = torch.softmax(outputs,dim=1)
        loss = criterion(outputs, masks)
        dice1, dice2, dice3 = cal_dice(outputs, masks)
        pbar.set_postfix(loss=f"{loss.item():.3f}")

        running_loss += loss.item()
        dice1_train += dice1.item()
        dice2_train += dice2.item()
        dice3_train += dice3.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss = running_loss / len(train_loader)
    dice1 = dice1_train / len(train_loader)
    dice2 = dice2_train / len(train_loader)
    dice3 = dice3_train / len(train_loader)
    return {'loss': loss, 'dice1': dice1, 'dice2': dice2, 'dice3': dice3}


def val_loop(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0
    dice1_val = 0
    dice2_val = 0
    dice3_val = 0
    pbar = tqdm(val_loader, desc='Validation: ', colour='#9868a8', unit='sample')
    with torch.no_grad():
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            # outputs = torch.softmax(outputs,dim=1)

            loss = criterion(outputs, masks)
            dice1, dice2, dice3 = cal_dice(outputs, masks)

            running_loss += loss.item()
            dice1_val += dice1.item()
            dice2_val += dice2.item()
            dice3_val += dice3.item()
            # pbar.desc = "loss:{:.3f} dice1:{:.3f} dice2:{:.3f} dice3:{:.3f} ".format(loss,dice1,dice2,dice3)
            pbar.set_postfix(loss=f"{loss:.3f}", dice1=f'{dice1:.3f}', dice2=f"{dice2:.3f}", dice3=f"{dice3:.3f}")
    loss = running_loss / len(val_loader)
    dice1 = dice1_val / len(val_loader)
    dice2 = dice2_val / len(val_loader)
    dice3 = dice3_val / len(val_loader)
    return {'loss': loss, 'dice1': dice1, 'dice2': dice2, 'dice3': dice3}


def train(model, optimizer, scheduler, criterion, train_loader,
          val_loader, epochs, device, train_log, valid_loss_min=999.0):
    for e in range(epochs):

        # train for epoch
        train_metrics = train_loop(model, optimizer, scheduler, criterion, train_loader, device, e)
        # eval for epoch
        val_metrics = val_loop(model, criterion, val_loader, device)
        scheduler.step(train_metrics['loss'])

        info1 = "Epoch:[{}/{}] train_loss: {:.3f} valid_loss: {:.3f} ".format(e+1, epochs, train_metrics["loss"], val_metrics["loss"])
        info2 = "Train--ET: {:.3f} TC: {:.3f} WT: {:.3f} ".format(train_metrics['dice1'], train_metrics['dice2'], train_metrics['dice3'])
        info3 = "Valid--ET: {:.3f} TC: {:.3f} WT: {:.3f} ".format(val_metrics['dice1'], val_metrics['dice2'], val_metrics['dice3'])
        print(info1)
        print(info2)
        print(info3)
        with open(train_log, 'a+') as f:
            f.write(info1 + '\n' + info2 + ' ' + info3 + '\n')

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict()}

        if val_metrics['loss'] <= valid_loss_min:
            valid_loss_min = val_metrics['loss']
            torch.save(save_file, f'results/{MODEL_NAME}/{MODEL_NAME}.pth')
        else:
            torch.save(save_file, os.path.join(args.save_path, 'checkpoint{}.pth'.format(e+1)))
    print("Finished Training!")


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(args.seed)  # 为所有的GPU设置种子，以使得结果是确定的

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data info
    patch_size = (128, 128, 64)
    train_dataset = BraTS2021(args.data_path, args.train_txt, transform=transforms.Compose([
        RandomRotFlip(),
        CenterCrop(patch_size),
        GaussianNoise(p=0.1),
        ToTensor()
    ]))
    val_dataset = BraTS2021(args.data_path, args.valid_txt, transform=transforms.Compose([
        CenterCrop(patch_size),
        ToTensor()
    ]))
    # test_dataset = BraTS2021(args.data_path, args.test_txt, transform=transforms.Compose([
    #     CenterCrop(patch_size),
    #     ToTensor()
    # ]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=8,   # num_worker=4
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False,
                            pin_memory=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=12, shuffle=False,
    #                          pin_memory=True)

    print("using {} device.".format(device))
    print("using {} images for training, {} images for validation.".format(len(train_dataset), len(val_dataset)))

    # 1-坏疽(NT,necrotic tumor core),2-浮肿区域(ED,peritumoral edema),4-增强肿瘤区域(ET,enhancing tumor)
    # 评价指标：ET(label4),TC(label1+label4),WT(label1+label2+label4)
    print(f"using {MODEL_NAME} for training.")
    if MODEL_NAME == 'MyNet':
        model = MyNet(in_channels=4, num_classes=4).to(device)
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
    elif MODEL_NAME == 'baseline':
        model = Baseline(in_channels=4, num_classes=4).to(device)
    elif MODEL_NAME == 'baselineMRF':
        model = BaselineMRF(in_channels=4, num_classes=4).to(device)
    elif MODEL_NAME == 'baselineRSA':
        model = BaselineRSA(in_channels=4, num_classes=4).to(device)

    criterion = Loss(n_classes=4, weight=torch.tensor([0.25, 0.25, 0.25, 0.25])).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.01, min_lr=1e-8, patience=2)

    # 加载训练模型
    if os.path.exists(args.weights):
        weight_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(weight_dict['model'])
        optimizer.load_state_dict(weight_dict['optimizer'])
        print('Successfully loading checkpoint.')

    train(model, optimizer, scheduler, criterion, train_loader, val_loader, args.epochs, device, train_log=args.train_log)

    # metrics1 = val_loop(model, criterion, train_loader, device)
    # metrics2 = val_loop(model, criterion, val_loader, device)
    # metrics3 = val_loop(model, criterion, test_loader, device)

    # 最后再评价一遍所有数据，注意，这里使用的是训练结束的模型参数
    # print("Valid -- loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics2['loss'], metrics2['dice1'], metrics2['dice2'], metrics2['dice3']))
    # print("Test  -- loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics3['loss'], metrics3['dice1'], metrics3['dice2'], metrics3['dice3']))


if __name__ == '__main__':
    # MODEL_NAME = 'ZhaoXiangYu'
    MODEL_NAME = 'MyNet'
    # MODEL_NAME = 'PeirisHimashi'
    # MODEL_NAME = 'Yi_Ding'
    # MODEL_NAME = 'Zhengrong_Luo'
    # MODEL_NAME = 'LiuLiangLiang'
    # MODEL_NAME = 'TheophrasteHenry'
    # MODEL_NAME = 'ChenChen'
    # MODEL_NAME = 'lslam'
    # MODEL_NAME = 'baseline'
    # MODEL_NAME = 'baselineMRF'
    # MODEL_NAME = 'baselineRSA'
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=0.002)
    parser.add_argument('--data_path', type=str, default='../dataset/brats2021/data')
    parser.add_argument('--train_txt', type=str, default='../dataset/brats2021/train.txt')
    parser.add_argument('--valid_txt', type=str, default='../dataset/brats2021/valid.txt')
    parser.add_argument('--train_log', type=str, default=f'results/{MODEL_NAME}/{MODEL_NAME}.txt')
    parser.add_argument('--weights', type=str, default=f'results/{MODEL_NAME}/{MODEL_NAME}.pth')
    parser.add_argument('--save_path', type=str, default=f'checkpoint/{MODEL_NAME}')

    args = parser.parse_args()

    if not os.path.exists(f'results/{MODEL_NAME}'):
        os.makedirs(f'results/{MODEL_NAME}')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    main(args)
