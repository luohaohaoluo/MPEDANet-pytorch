import os
import pathlib
from sklearn.model_selection import train_test_split

# 预处理输出地址
train_data_path = "../dataset/brats2021/data"
# validation_data_path = "../dataset/brats2020/val_data/data"

train_and_test_ids = os.listdir(train_data_path)
# train_and_test_ids.remove('BraTS20_Training_355')  # 去掉已损坏的355
print(train_and_test_ids)
# val_ids = os.listdir(validation_data_path)

train_ids, val_test_ids = train_test_split(train_and_test_ids, test_size=0.2, random_state=21)
val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5, random_state=21)
print("Using {} images for training, {} images for validation,\
{} images for testing.".format(len(train_ids), len(val_ids), len(test_ids)))

# print("Using {} images for training, {} images for validation.".format(len(train_ids), len(val_ids)))
train_ids.sort()
val_ids.sort()
test_ids.sort()
print(train_ids)

with open('../dataset/brats2021/train.txt', 'w') as f:
    f.write('\n'.join(train_ids))

with open('../dataset/brats2021/valid.txt', 'w') as f:
    f.write('\n'.join(val_ids))

with open('../dataset/brats2021/test.txt', 'w') as f:
    f.write('\n'.join(test_ids))
