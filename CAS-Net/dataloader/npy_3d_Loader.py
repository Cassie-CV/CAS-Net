from __future__ import print_function, division
import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import warnings
import numpy as np
from scipy import ndimage


warnings.filterwarnings('ignore')


def load_dataset(root_dir, folder, train=True):
    images_path = os.path.join(root_dir, 'img')
    groundtruth_path = os.path.join(root_dir, 'mask')
    print("folder", folder)
    # images_path = os.path.join(root_dir,  'img')
    # groundtruth_path = os.path.join(root_dir, 'mask')
    # i=0
    folder_file = './datasets/' + folder

    # if self.train_type in ['train', 'validation', 'test']:
    if train:
        # this is for cross validation
        with open(os.path.join(folder_file, folder_file.split('/')[-1] + '_' + 'train' + '.list'),
                  'r') as f:
            image_list = f.readlines()
    else:
        with open(os.path.join(folder_file, folder_file.split('/')[-1] + '_' + 'validation' + '.list'),
                  'r') as f:
            image_list = f.readlines()
    image_list = [item.replace('\n', '') for item in image_list]
    images = [os.path.join(root_dir, 'img', x) for x in image_list]
    groundtruth = [os.path.join(root_dir, 'mask', x) for x in image_list]
    return images, groundtruth


def resize3d(img3d, reshape):
    zoom_seq = np.array(reshape, dtype='float') / np.array(img3d.shape, dtype='float')
    ret = ndimage.interpolation.zoom(img3d, zoom_seq, order=1, prefilter=False)
    return ret.astype(img3d.dtype)


class Data(Dataset):
    def __init__(self,
                 root_dir,
                 folder,
                 input_shape,
                 train=True,
                 rotate=40,
                 flip=True,
                 random_crop=True,
                 scale1=512):

        self.root_dir = root_dir
        self.folder = folder
        self.input_shape = input_shape
        self.train = train
        self.rotate = rotate
        self.flip = flip
        self.random_crop = random_crop
        self.transform = transforms.ToTensor()
        self.resize = scale1
        self.images, self.groundtruth = load_dataset(self.root_dir, self.folder, self.train)

    def __len__(self):
        return len(self.images)

    def RandomCrop(self, image, label, crop_factor=(0, 0, 0)):
        """
        Make a random crop of the whole volume
        :param image:
        :param label:
        :param crop_factor: The crop size that you want to crop
        :return:
        """
        # print(crop_factor)
        w, h, d = image.shape
        # print(image.shape,h - crop_factor[1])

        try:
            y = random.randint(0, h - crop_factor[1])
        except:
            y = 0
        try:
            x = random.randint(0, d - crop_factor[2])
        except:
            x = 0

        if w <= crop_factor[0]:
            image = image[:, y:y + crop_factor[1], x:x + crop_factor[2]]
            label = label[:, y:y + crop_factor[1], x:x + crop_factor[2]]
            image = resize3d(image, self.input_shape)
            label = resize3d(label, self.input_shape)
        else:
            z = random.randint(0, w - crop_factor[0])
            image = image[z:z + crop_factor[0], y:y + crop_factor[1], x:x + crop_factor[2]]
            label = label[z:z + crop_factor[0], y:y + crop_factor[1], x:x + crop_factor[2]]
            if y == 0 or x == 0:
                image = resize3d(image, self.input_shape)
                label = resize3d(label, self.input_shape)
        return image, label

    def __getitem__(self, idx):
        img_path = self.images[idx]
        gt_path = self.groundtruth[idx]

        image = np.load(img_path).astype(np.float32)  # sitk.ReadImage(img_path)
        label = np.load(gt_path)
        label[label > 0] = 255
        label = label.astype(np.int64)
        try:
            image, label = self.RandomCrop(image, label, crop_factor=self.input_shape)  # [z,y,x]
        except:
            print("----------------")
            image, label = self.RandomCrop(image, label, crop_factor=self.input_shape)  # [z,y,x]

        if self.train:
            # image, label = self.RandomCrop(image, label, crop_factor=self.input_shape)  # [z,y,x]
            seed_num = np.random.randint(0, 10)
            if seed_num in [0, 1, 2, 4, 5]:
                sigma = np.random.choice(np.array([0.1, 0.3, 0.5, 0.7, 1]), 1)[0]
                image = ndimage.gaussian_filter(image, sigma=sigma)
            image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
            label = torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0)


        else:
            image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
            label = torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0)

        image = image / 255
        label = label // 255

        return image, label