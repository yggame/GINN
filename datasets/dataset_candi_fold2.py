import os
import pdb
import numpy as np
import h5py
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from sklearn.model_selection import KFold, train_test_split
import SimpleITK as sitk


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


def read_nii_img(path):
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)
    return img

def saveimg(img, path):
    img = sitk.GetImageFromArray(img)
    return sitk.WriteImage(img, path)


class Candi_dataset_fold(Dataset):
    def __init__(self,  base_dir, n_fold, fold_num, dim, split, random_seed, transform=None):
        self.transform = transform  # using transform in torch!
        self.dim = dim
        self.split = split
        assert dim==2 or dim==3
        
        file_name = os.listdir(os.path.join(base_dir, "imagesTr"))
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=random_seed)
        fl_train_idx, fl_val_idx = list(kf.split(file_name))[fold_num] #0-4

        self.fl_train = np.array(file_name)[fl_train_idx]
        self.fl_val = np.array(file_name)[fl_val_idx]

        self.example = []
        if split == "train":
            if dim == 3:
                for file in self.fl_train:
                    img = read_nii_img(os.path.join(base_dir, "imagesTr", str(file)))
                    seg = read_nii_img(os.path.join(base_dir, "labelsTr", file[:-12]+".nii.gz"))

                    self.example.append((img, seg, file, -1))
            else:
                for file in self.fl_train:
                    img = read_nii_img(os.path.join(base_dir, "imagesTr", str(file)))
                    seg = read_nii_img(os.path.join(base_dir, "labelsTr", file[:-12]+".nii.gz"))

                    d, h, w = img.shape
                    for slice in range(d):
                        self.example.append((img[slice], seg[slice], file, slice))
        
        else:
            # if dim == 3:
            for file in self.fl_val:
                img = read_nii_img(os.path.join(base_dir, "imagesTr", str(file)))
                seg = read_nii_img(os.path.join(base_dir, "labelsTr", file[:-12]+".nii.gz"))

                self.example.append((img, seg, file, -1))

        self.data_dir = base_dir
    
    def __getitem__(self, idx):
        image, label, file, slice = self.example[idx]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.example[idx][2][:-12]
        return sample
    
    def __len__(self):
        return len(self.example)
