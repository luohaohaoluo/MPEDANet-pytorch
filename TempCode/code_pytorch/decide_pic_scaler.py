import nibabel as nib
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


if __name__ == '__main__':
    TRAIN_DATASET_PATH = '../dataset/brats2021/data/'
    case = '01512'

    test_image_flair = sitk.GetArrayFromImage(sitk.ReadImage(TRAIN_DATASET_PATH + f'BraTS2021_{case}/BraTS2021_{case}_flair.nii.gz')).transpose(1, 2, 0)

    test_image_t1 = sitk.GetArrayFromImage(sitk.ReadImage(TRAIN_DATASET_PATH + f'BraTS2021_{case}/BraTS2021_{case}_t1.nii.gz')).transpose(1, 2, 0)

    test_image_t1ce = sitk.GetArrayFromImage(sitk.ReadImage(TRAIN_DATASET_PATH + f'BraTS2021_{case}/BraTS2021_{case}_t1ce.nii.gz')).transpose(1, 2, 0)

    test_image_t2 = sitk.GetArrayFromImage(sitk.ReadImage(TRAIN_DATASET_PATH + f'BraTS2021_{case}/BraTS2021_{case}_t2.nii.gz')).transpose(1, 2, 0)

    test_mask = sitk.GetArrayFromImage(sitk.ReadImage(TRAIN_DATASET_PATH + f'BraTS2021_{case}/BraTS2021_{case}_seg.nii.gz')).transpose(1, 2, 0)
    print(test_image_flair.shape)
    print(np.unique(test_mask))
    print(test_mask.shape)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 10))
    slice_w = 45
    ax1.imshow(test_image_flair[:, :, slice_w], cmap='gray')
    ax1.set_title('Image flair')
    ax2.imshow(test_image_t1[:, :, slice_w], cmap='gray')
    ax2.set_title('Image t1')
    ax3.imshow(test_image_t1ce[:, :, slice_w], cmap='gray')
    ax3.set_title('Image t1ce')
    ax4.imshow(test_image_t2[:, :, slice_w], cmap='gray')
    ax4.set_title('Image t2')

    mask_segmentation = test_mask[:, :, slice_w]
    color_segmentation = np.zeros((240, 240, 3))
    color_segmentation[mask_segmentation == 1] = [248, 2, 3]  # Red (necrotic tumor core)
    color_segmentation[mask_segmentation == 2] = [9, 122, 6]  # Green (peritumoral edematous/invaded tissue)
    color_segmentation[mask_segmentation == 4] = [255, 255, 21]  # Yellow (enhancing tumor)

    ax5.imshow(color_segmentation.astype('uint8'), cmap='gray')
    # ax5.imshow(mask_segmentation, cmap='gray')
    ax5.set_title('Mask')
    plt.show()


