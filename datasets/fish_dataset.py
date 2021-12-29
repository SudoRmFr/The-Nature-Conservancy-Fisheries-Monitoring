""" CUB-200-2011 (Bird) Dataset"""
import os
import pdb
from PIL import Image
from torch.utils.data import Dataset
from utils import get_transform
from typing import List

DATAPATH = './the-nature-conservancy-fisheries-monitoring'
image_path = {}
image_label = {}


class FishDataset(Dataset):
    """
    # Description:
        There are 8 classes in total.
        Each of them is assigned an id number, which is given below in "CLASS_NAME_TO_ID" and "CLASS_ID_TO_NAME"

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, idx):        returns an image
            idx:                       the index of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """
    # Get class id by class name
    # We need this since we use id when training instead of a string of name
    CLASS_NAME_TO_ID = {
        'ALB': 0,
        'BET': 1,
        'DOL': 2,
        'LAG': 3,
        'NoF': 4,
        'OTHER': 5,
        'SHARK': 6,
        'YFT': 7
    }
    # Get class name by class id
    CLASS_ID_TO_NAME = {
        0: 'ALB',
        1: 'BET',
        2: 'DOL',
        3: 'LAG',
        4: 'NoF',
        5: 'OTHER',
        6: 'SHARK',
        7: 'YFT'
    }
    # Image format
    IMG_FORMAT = 'jpg'

    def __init__(self, fold, phase='train', resize=500):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize
        self.fold = fold
        self.image_id = []
        self.num_classes = 8

        self.names = []
        self.classes = []

        if fold in [1, 2, 3, 4]:
            train_txt_name = f'train_fold_{fold}.txt'
            valid_txt_name = f'valid_fold_{fold}.txt'
        else:
            train_txt_name = f'train.txt'
            valid_txt_name = f'valid.txt'

        print(train_txt_name)
        print(valid_txt_name)

        if phase == 'train':
            with open(os.path.join(DATAPATH, train_txt_name), 'r') as f:
                for line in f.readlines():
                    name, cls = line.strip().split(' ')
                    self.names.append(name)
                    self.classes.append(int(cls))
        elif phase == 'val':
            with open(os.path.join(DATAPATH, valid_txt_name), 'r') as f:
                for line in f.readlines():
                    name, cls = line.strip().split(' ')
                    self.names.append(name)
                    self.classes.append(int(cls))
        else:  # test
            with open(os.path.join(DATAPATH, 'test.txt'), 'r') as f:
                for line in f.readlines():
                    name = line.strip()
                    self.names.append(name)
        # transform
        self.transform = get_transform(self.resize, self.phase)

    def __getitem__(self, idx):
        if self.phase in ['train', 'val']:
            name = self.names[idx]
            cls = self.classes[idx]
            img = Image.open(os.path.join(DATAPATH, 'train', name)).convert('RGB')
            img = self.transform(img)
            return img, cls

        else:  # test
            name = self.names[idx]
            img = Image.open(os.path.join(DATAPATH, 'test', name)).convert('RGB')
            img = self.transform(img)
            return img, name

    def __len__(self):
        return len(self.names)


if __name__ == '__main__':
    ds = FishDataset('train')
    print(len(ds))
    for i in range(0, 10):
        image, label = ds[i]
        print(image.shape, label)
