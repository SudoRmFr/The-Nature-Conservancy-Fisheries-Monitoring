from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

"""
There are 8 classes in total.
Each of them is assigned an id number, which is given below in "CLASS_NAME_TO_ID" and "CLASS_ID_TO_NAME"
"""
VALID_SIZE = 0.1
RANDOM_STATE = 666666
DATAPATH = './the-nature-conservancy-fisheries-monitoring'

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


def get_x_names_y_ids(img_root: str) -> (List[str], List[int]):
    x_names = []
    y_ids = []
    for root, dirs, files in os.walk(img_root):
        cls_name = root.split('/')[-1]
        cls_relpath = os.path.relpath(root, img_root)
        for file in files:
            if file.split('.')[-1] == IMG_FORMAT:
                x_names.append(os.path.join(cls_relpath, file))
                y_ids.append(CLASS_NAME_TO_ID[cls_name])
    return x_names, y_ids


def get_test_names(te_root: str):
    names = []
    for root, dirs, files in os.walk(te_root):
        for file in files:
            relpath = os.path.relpath(root, te_root)
            if file.split('.')[-1] == IMG_FORMAT:
                names.append(os.path.join(relpath, file))
    return names


def make_txt(tr_root: str,
             te_root: str,
             valid_size: float,
             random_state: int,
             stratify: bool = True):
    x, y = get_x_names_y_ids(tr_root)
    t_x, v_x, t_y, v_y = train_test_split(x, y,
                                          test_size=valid_size,
                                          random_state=random_state,
                                          shuffle=True,
                                          stratify=y if stratify else None)
    # Make train txt
    with open(os.path.join(DATAPATH, 'train.txt'), 'w') as f:
        for name, label in zip(t_x, t_y):
            f.write(f'{name} {label}\n')
    # Make valid txt
    with open(os.path.join(DATAPATH, 'valid.txt'), 'w') as f:
        for name, label in zip(v_x, v_y):
            f.write(f'{name} {label}\n')
    # Make test txt
    test_names = get_test_names(te_root)
    with open(os.path.join(DATAPATH, 'test.txt'), 'w') as f:
        for name in test_names:
            f.write(f'{name}\n')


def make_4_fold_txt(tr_root: str,
                    te_root: str,
                    random_state: int):
    x, y = get_x_names_y_ids(tr_root)
    x = np.array(x)
    y = np.array(y)

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    skf.get_n_splits(x, y)

    print(skf)
    for fold, (train_index, test_index) in enumerate(skf.split(x, y)):
        train_index = train_index.astype(int)
        test_index = test_index.astype(int)
        t_x, v_x = x[train_index], x[test_index]
        t_y, v_y = y[train_index], y[test_index]
        # Make train txt
        with open(os.path.join(DATAPATH, f'train_fold_{fold + 1}.txt'), 'w') as f:
            for name, label in zip(t_x, t_y):
                f.write(f'{name} {label}\n')
        # Make valid txt
        with open(os.path.join(DATAPATH, f'valid_fold_{fold + 1}.txt'), 'w') as f:
            for name, label in zip(v_x, v_y):
                f.write(f'{name} {label}\n')
    # Make test txt
    test_names = get_test_names(te_root)
    with open(os.path.join(DATAPATH, 'test.txt'), 'w') as f:
        for name in test_names:
            f.write(f'{name}\n')


if __name__ == '__main__':
    TR_ROOT = os.path.join(DATAPATH, 'train')
    TE_ROOT = os.path.join(DATAPATH, 'test')
    make_txt(tr_root=TR_ROOT,
             te_root=TE_ROOT,
             valid_size=VALID_SIZE,
             random_state=RANDOM_STATE)
    make_4_fold_txt(tr_root=TR_ROOT,
                    te_root=TE_ROOT,
                    random_state=RANDOM_STATE)
