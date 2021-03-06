"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import pickle
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.mypath import MyPath
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import tensorflow_datasets as tfds
import cv2


class Omniglot(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root=MyPath.db_root_dir('omniglot'), train=True, transform=None, 
                    download=True):

        super(Omniglot, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        if train:
            self.split = 'train'
        else:
            self.split = 'test'
        ds, info = tfds.load("omniglot", split=self.split, shuffle_files=True, with_info = True)
        ds_np = tfds.as_numpy(ds)
        self.classes = info.features["label"].names

        self.data = []
        self.targets = []
        #i=0

        # now load the picked numpy arrays
        for ex in ds_np:
            _img = cv2.resize(ex["image"], (32, 32))
            self.data.append(_img)
            self.targets.append(ex["label"])
            del _img
            #i+=1
            #if i == 1000:
            #    break

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)
        class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'class_name': class_name}}
        
        return out

    def get_image(self, index):
        img = self.data[index]
        return img
        
    def __len__(self):
        return len(self.data)
