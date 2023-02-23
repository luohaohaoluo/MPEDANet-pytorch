import os
import torch
import random
import numpy as np
import h5py
import matplotlib.pyplot as plt
import SimpleITK as sitk

from torchvision import transforms
from torch.utils.data import Dataset

modalities = ('_flair', '_t1ce', '_t1', '_t2')

START = 45
END = 109
# 45 ~ 108 一共 64


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # w,h 其实在这里并没有什么区别
        (c, w, h, d) = image.shape
        # randint(0, 240 - 160)
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], START:END]
        image = image[:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], START:END]
        # label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        # image = image[:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        (c, w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], START:END]
        image = image[:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], START:END]
        # label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1+self.output_size[2]]
        # image = image[:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1+self.output_size[2]]
        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.stack([np.rot90(x, k) for x in image], axis=0)
        label = np.rot90(label, k)
        axis = np.random.randint(1, 4)
        # image.shape: torch.Size([4, 160, 160, 64])
        # label.shape: torch.Size([160, 160, 64])

        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis - 1).copy()

        return {'image': image, 'label': label}


def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample


class GaussianNoise(object):
    def __init__(self, noise_variance=(0, 0.1), p=0.5):
        self.prob = p
        self.noise_variance = noise_variance

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if np.random.uniform() < self.prob:
            image = augment_gaussian_noise(image, self.noise_variance)
        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}


class BraTS2021(Dataset):
    def __init__(self, data_path, file_path, transform=None):
        with open(file_path, 'r') as f:
            self.paths = [os.path.join(data_path, x.strip()) for x in f.readlines()]
        # print(self.paths)
        self.transform = transform

    def __getitem__(self, item):

        label = sitk.GetArrayFromImage(sitk.ReadImage(self.paths[item] + '/' + self.paths[item][-15:] + '_seg.nii.gz')).transpose(1, 2, 0)
        # print(label.shape)
        # 堆叠四种模态的图像，4 x (H,W,D) -> (4,H,W,D)
        images = np.stack(
            [sitk.GetArrayFromImage(sitk.ReadImage(self.paths[item] + '/' + self.paths[item][-15:] + modal + '.nii.gz')).transpose(1, 2, 0) for modal in
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
        # print(image.shape)
        sample = {'image': images, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample['image'], sample['label']

    def __len__(self):
        return len(self.paths)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

# class BraTS2021(Dataset):
#     def __init__(self, data_path, file_path, transform=None):
#         with open(file_path, 'r') as f:
#             self.paths = [os.path.join(data_path, x.strip()) for x in f.readlines()]
#         # print(self.paths)
#         self.transform = transform
#
#     def __getitem__(self, item):
#
#         label = sitk.GetArrayFromImage(sitk.ReadImage(self.paths[item] + '/' + self.paths[item][-15:] + '_seg.nii.gz')).transpose(1, 2, 0)
#         # print(label.shape)
#         # 堆叠四种模态的图像，4 x (H,W,D) -> (4,H,W,D)
#         images = np.stack(
#             [sitk.GetArrayFromImage(sitk.ReadImage(self.paths[item] + '/' + self.paths[item][-15:] + modal + '.nii.gz')).transpose(1, 2, 0) for modal in
#              modalities],
#             0)  # [240,240,155]
#         # 数据类型转换
#         label = label.astype(np.uint8)
#         images = images.astype(np.float32)
#
#         # 对第一个通道求和，如果四个模态都为0，则标记为背景(False)
#
#         # [0,1,2,4] -> [0,1,2,3]
#         label[label == 4] = 3
#         # print(image.shape)
#         sample = {'image': images, 'label': label}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample['image'], sample['label']
#
#     def __len__(self):
#         return len(self.paths)
#
#     def collate(self, batch):
#         return [torch.cat(v) for v in zip(*batch)]


if __name__ == '__main__':
    np.random.seed(21)
    data_path = "../dataset/brats2021/data"
    test_txt = "../dataset/brats2021/train.txt"
    size = (128, 128, 64)
    test_set = BraTS2021(data_path, test_txt, transform=transforms.Compose([
        RandomRotFlip(),
        CenterCrop(size),
        GaussianNoise(p=0.1),
        ToTensor()
    ]))
    d1 = test_set[45]
    image, label = d1
    print("image.shape:", image.shape)
    print("label.shape:", label.shape)
    print(np.unique(label))

    slice_w = 40
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))
    ax1.imshow(image[0, :, :, slice_w], cmap='gray')
    ax1.set_title('Image flair')
    ax2.imshow(image[1, :, :, slice_w], cmap='gray')
    ax2.set_title('Image t1ce')
    ax3.imshow(image[2, :, :, slice_w], cmap='gray')
    ax3.set_title('Image t1')
    ax4.imshow(image[3, :, :, slice_w], cmap='gray')
    ax4.set_title('Image t2')

    mask_segmentation = label[:, :, slice_w]
    color_segmentation = np.zeros((size[0], size[1], 3))
    color_segmentation[mask_segmentation == 1] = [255, 0, 0]  # Red (necrotic tumor core)
    color_segmentation[mask_segmentation == 2] = [23, 102, 17]  # Green (peritumoral edematous/invaded tissue)
    color_segmentation[mask_segmentation == 3] = [250, 246, 45]  # Yellow (enhancing tumor)
    ax5.imshow(color_segmentation.astype('uint8'))
    # ax2.imshow(color_segmentation.astype('uint8'))
    # ax5.imshow(label[:, :, slice_w], cmap='gray')
    ax5.set_title('Mask')

    plt.show()

