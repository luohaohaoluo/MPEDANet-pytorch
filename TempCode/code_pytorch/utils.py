import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import operator
from einops import rearrange
from medpy import metric
import SimpleITK as sitk


"""
该文件包括 ：
（1）训练需要的 dice_loss dice
（2）评估需要的 specificity sensitivity Hausdorff_distance
"""


def hausdorff_distance(pre, label):  # Input be like (Depth,width,height)
    x = torch.round(pre.float())
    y = torch.round(label.float())

    distance_matrix = torch.cdist(x, y, p=2)  # p=2 means Euclidean Distance
    # print(distance_matrix.min(2))
    value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
    value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]

    value = torch.cat((value1, value2), dim=1)
    # print(value.max(1)[0])
    length = (value.max(1)[0] != 0).sum()
    return sum(value.max(1)[0]) / length


def hausdorff_distance_WT(pre, label):
    pre = torch.cat([pre[0, 1, ...], pre[0, 2, ...], pre[0, 3, ...]], dim=1)
    label = torch.cat([label[0, 1, ...], label[0, 2, ...], label[0, 3, ...]], dim=1)
    return hausdorff_distance(pre, label)


def hausdorff_distance_ET(pre, label):
    pre = pre[0, 3, ...]
    label = label[0, 3, ...]
    return hausdorff_distance(pre, label)


def hausdorff_distance_TC(pre, label):
    pre = torch.cat([pre[0, 1, ...], pre[0, 3, ...]], dim=1)
    label = torch.cat([label[0, 1, ...], label[0, 3, ...]], dim=1)
    return hausdorff_distance(pre, label)


def sensitivity(predict, target):
    result = metric.binary.sensitivity(predict, target)
    return result


def sensitivity_WT(predict, target):
    predict = np.concatenate([predict[:, 1, ...], predict[:, 2, ...], predict[:, 3, ...]], axis=1)
    target = np.concatenate([target[:, 1, ...], target[:, 2, ...], target[:, 3, ...]], axis=1)
    result = sensitivity(predict, target)

    return result


def sensitivity_ET(predict, target):
    predict = predict[:, 3, ...]
    target = target[:, 3, ...]

    result = sensitivity(predict, target)

    return result


def sensitivity_TC(predict, target):
    predict = np.concatenate([predict[:, 1, ...], predict[:, 3, ...]], axis=1)
    target = np.concatenate([target[:, 1, ...], target[:, 3, ...]], axis=1)

    result = sensitivity(predict, target)

    return result


def specificity(predict, target):
    result = metric.binary.specificity(predict, target)
    return result


def specificity_WT(predict, target):
    predict = np.concatenate([predict[:, 1, ...], predict[:, 2, ...], predict[:, 3, ...]], axis=1)
    target = np.concatenate([target[:, 1, ...], target[:, 2, ...], target[:, 3, ...]], axis=1)
    result = specificity(predict, target)

    return result


def specificity_ET(predict, target):
    predict = predict[:, 3, ...]
    target = target[:, 3, ...]

    result = specificity(predict, target)

    return result


def specificity_TC(predict, target):
    predict = np.concatenate([predict[:, 1, ...], predict[:, 3, ...]], axis=1)
    target = np.concatenate([target[:, 1, ...], target[:, 3, ...]], axis=1)

    result = specificity(predict, target)

    return result


def Dice(output, target, eps=1e-6):
    inter = torch.sum(output * target, dim=(1, 2, -1))
    union = torch.sum(output, dim=(1, 2, -1)) + torch.sum(target, dim=(1, 2, -1)) + eps
    x = 2 * inter / union
    dice = torch.mean(x)
    return dice


def cal_dice(output, target):
    '''
    output: (b, num_class, d, h, w)  target: (b, d, h, w)
    dice1(ET): label4
    dice2(TC): label1 + label4
    dice3(WT): label1 + label2 + label4
    注,这里的label4已经被替换为3
    '''
    output = torch.argmax(output, dim=1)
    dice1 = Dice((output == 3).float(), (target == 3).float())
    dice2 = Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    dice3 = Dice((output != 0).float(), (target != 0).float())

    return dice1, dice2, dice3


class Loss(nn.Module):
    def __init__(self, n_classes, weight=None, alpha=0.5):
        "dice_loss_plus_cetr_weighted"
        super(Loss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight.cuda()
        # self.weight = weight
        self.alpha = alpha

    def forward(self, input, target):
        # print(torch.unique(target))
        # input: (1, 4, 160, 160, 64)
        # target: (1, 160, 160, 64)
        smooth = 1e-6

        input1 = F.softmax(input, dim=1)
        target1 = F.one_hot(target, self.n_classes)
        # input1: torch.Size([1, 4, 160, 160, 64])
        # target1: torch.Size([1, 160, 160, 64, 4])

        input1 = rearrange(input1, 'b n h w s -> b n (h w s)')
        target1 = rearrange(target1, 'b h w s n -> b n (h w s)')
        # input1: torch.Size([1, 4, (160*160*64)])
        # target1: torch.Size([1, 4, (160*160*64)])

        input1 = input1[:, 1:, :]
        target1 = target1[:, 1:, :].float()
        # input1: torch.Size([1, 3, (160*160*64)])
        # target1: torch.Size([1, 3, (160*160*64)])

        # 以batch为单位计算loss和dice_loss，据说训练更稳定，那我试试
        inter = torch.sum(input1 * target1)
        union = torch.sum(input1) + torch.sum(target1) + smooth
        dice = 2.0 * inter / union

        loss = F.cross_entropy(input, target, weight=self.weight)

        total_loss = (1 - self.alpha) * loss + (1 - dice) * self.alpha

        return total_loss


if __name__ == '__main__':
    torch.manual_seed(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losser = Loss(n_classes=4, weight=torch.tensor([0.2, 0.3, 0.25, 0.25])).to(device)
    x = torch.randn((2, 4, 16, 16, 16)).to(device)
    y = torch.randint(0, 4, (2, 16, 16, 16)).to(device)
    print(losser(x, y))
    print(cal_dice(x, y))
