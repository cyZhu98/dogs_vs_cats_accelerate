import numpy
import os
from PIL import Image
import cv2
import torch
import torch.utils.data as data
import random

from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip, Resize, RandomResizedCrop, CenterCrop
from albumentations.pytorch import ToTensorV2

from config import return_args

args = return_args()
random.seed(args.seed)
train_dir = os.path.join(args.DIR, 'train') + '/'
test_dir = os.path.join(args.DIR, 'test') + '/'
train_images = [train_dir+i for i in os.listdir(train_dir)]
test_images = [test_dir+i for i in os.listdir(test_dir)]
print('original training images number : ', len(train_images))
print('test images number : ', len(test_images))

# train_dogs = [train_dir+i for i in os.listdir(train_dir) if 'dog' in i]
# train_cats = [train_dir+i for i in os.listdir(train_dir) if 'cat' in i]
# original_train_images = train_dogs[:12000] + train_cats[:12000]
random.shuffle(train_images)

partition = int(args.split_ratio * len(train_images))
val_images = train_images[partition:]
train_images = train_images[:partition]

train_transform = Compose([
    RandomResizedCrop(224, 224),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    ToTensorV2(p=1.0), ])

test_transform = Compose([
    Resize(256, 256),
    CenterCrop(224, 224),
    Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    ToTensorV2(p=1.0), ])


class ImageLoader(data.Dataset):
    def __init__(self, split, transform=None):
        super().__init__()
        images = f'{split}' + '_images'
        self.images = eval(images)
        self.transform = transform

    def __getitem__(self, index):
        root = self.images[index]
        img = cv2.imread(root)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        label = 1 if 'dog' in root else 0
        return img, label

    def __len__(self):
        return len(self.images)


def getLoader(args, split):
    if split in ['dev', 'valid', 'validation']:
        split = 'val'
    transform = train_transform if split == 'train' else test_transform
    dataset = ImageLoader(split, transform)
    dataLoader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=split ==
                                 'train', num_workers=args.workers, pin_memory=True, drop_last=split == 'train')
    return dataLoader


if __name__ == '__main__':
    print(train_images[-1])
    pass
