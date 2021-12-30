import os.path
from PIL import Image
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from typing import Optional
from tqdm import tqdm


# TODO: resize to 448x448 first and resize to 224x224 after mosaic

# ================ Tools ==================
def show_tensor_img(img: torch.Tensor):
    assert 3 <= len(img.shape) <= 4
    if len(img.shape) == 3:
        img = img.numpy().transpose(1, 2, 0)
        plt.imshow(img)
        plt.show()
    else:
        grid = utils.make_grid(img, nrow=4)
        grid = grid.numpy().transpose(1, 2, 0)
        plt.imshow(grid)
        plt.show()


def get_img_names(img_root):
    names = []
    for root, dirs, files in os.walk(img_root):
        for file in files:
            if '.jpg' in file:
                names.append(os.path.join(root, file))
    return names


def mosaic_aug(img_path: str,
               n_aug_per_img: int,
               split_range: tuple = (0.4, 0.6),
               combine_range: tuple = (0.4, 0.6),
               show: bool = False,
               save_path: Optional[str] = None):
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    img = Image.open(img_path)

    original_transform = transforms.Compose([transforms.Resize(size=(448, 448)), transforms.ToTensor()])

    # TRA = transforms.Compose([transforms.Resize(size=(448, 448)),
    #                           # Good for now
    #                           transforms.RandomHorizontalFlip(p=0.5),
    #                           transforms.RandomVerticalFlip(p=0.5),
    #                           transforms.RandomRotation(degrees=(0, 180)),
    #                           # Color
    #                           # transforms.ColorJitter(hue=.1),
    #                           # transforms.RandomAdjustSharpness(sharpness_factor=2., p=.5),
    #                           # transforms.RandomPosterize(bits=3, p=.5),
    #
    #                           # Shape
    #                           # transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
    #
    #                           transforms.ToTensor()])

    for idx in range(n_aug_per_img):
        # Split a img
        split_hor = split_range[0] + (split_range[1] - split_range[0]) * np.random.random()
        split_ver = split_range[0] + (split_range[1] - split_range[0]) * np.random.random()
        arr = original_transform(img).numpy().transpose(1, 2, 0)
        h, w, _ = arr.shape

        # tile 1, 2, 3, 4 => top-left, top-right, bottom-left, bottom-right
        # ============> x
        # y
        tile1_x = 0
        tile1_y = 0
        tile1_w = int(w * split_hor)
        tile1_h = int(h * split_ver)
        tile1 = arr[tile1_x: tile1_x + tile1_w, tile1_y:tile1_y + tile1_h, :]

        tile2_x = tile1_x + tile1_w
        tile2_y = 0
        tile2_h = tile1_h
        tile2 = arr[tile2_x:, tile1_y:tile2_y + tile2_h, :]

        tile3_x = 0
        tile3_y = tile1_y + tile1_h
        tile3_w = tile1_w
        tile3 = arr[tile3_x:tile3_w + tile3_w, tile3_y:, :]

        tile4_x = tile1_x + tile1_w
        tile4_y = tile1_y + tile1_h
        tile4 = arr[tile4_x:, tile4_y:, :]

        # Random arrange
        indices = np.array([0, 1, 2, 3])
        indices = np.random.permutation(indices)
        tiles = [tile1, tile2, tile3, tile4]
        new_tile1 = tiles[indices[0]]
        new_tile2 = tiles[indices[1]]
        new_tile3 = tiles[indices[2]]
        new_tile4 = tiles[indices[3]]

        def cv2_transform(cv2_img):  # transforms
            if np.random.random() < 0.5:
                cv2_img = cv2.flip(cv2_img, 1)
            if np.random.random() < 0.5:
                cv2_img = cv2.flip(cv2_img, 0)
            if np.random.random() < 0.5:
                cv2_img = cv2.rotate(cv2_img, cv2.ROTATE_90_CLOCKWISE)
            return cv2_img

        new_tile1 = cv2_transform(new_tile1)
        new_tile2 = cv2_transform(new_tile2)
        new_tile3 = cv2_transform(new_tile3)
        new_tile4 = cv2_transform(new_tile4)

        #
        combine_hor = combine_range[0] + (combine_range[1] - combine_range[0]) * np.random.random()
        combine_ver = combine_range[0] + (combine_range[1] - combine_range[0]) * np.random.random()

        new_tile1_h = int(h * combine_hor)
        new_tile1_w = int(w * combine_ver)
        new_tile2_h = int(h * combine_hor)
        new_tile2_w = w - int(w * combine_ver)
        new_tile3_h = h - int(h * combine_hor)
        new_tile3_w = int(w * combine_ver)
        new_tile4_h = h - int(h * combine_hor)
        new_tile4_w = w - int(w * combine_ver)

        new_tile1 = cv2.resize(new_tile1, dsize=(new_tile1_w, new_tile1_h)).transpose(2, 0, 1)
        new_tile2 = cv2.resize(new_tile2, dsize=(new_tile2_w, new_tile2_h)).transpose(2, 0, 1)
        new_tile3 = cv2.resize(new_tile3, dsize=(new_tile3_w, new_tile3_h)).transpose(2, 0, 1)
        new_tile4 = cv2.resize(new_tile4, dsize=(new_tile4_w, new_tile4_h)).transpose(2, 0, 1)

        new_img_top = np.concatenate((new_tile1, new_tile2), axis=2)
        new_img_bottom = np.concatenate((new_tile3, new_tile4), axis=2)
        new_img = np.concatenate((new_img_top, new_img_bottom), axis=1).transpose((1, 2, 0))

        if show is True:
            plt.imshow(new_img)
            plt.show()

        if save_path is not None:
            rand_no = np.random.randint(1, 999999999)
            save_name = os.path.join(save_path, f'{img_path.split("/")[-1]}_{rand_no:09d}.jpg')
            # cv2.imwrite(save_name, new_img)
            from matplotlib.image import imsave
            imsave(save_name, new_img)


# ================ Settings ===============
IMG_ROOT = '???/YFT'
SAVE_PATH = './aug/'
N_AUG_PER_IMG = 1

# ================== Init ===================
NAMES = get_img_names(img_root=IMG_ROOT)

for NAME in tqdm(NAMES, total=len(NAMES)):
    mosaic_aug(img_path=NAME,
               n_aug_per_img=N_AUG_PER_IMG,
               show=False,
               save_path=SAVE_PATH)

assert len(os.listdir(SAVE_PATH)) == len(NAMES) * N_AUG_PER_IMG, 'Random code may collide. Do it again.'
print('Done!')


#img = Image.open('???')
#img = Image.open('???')
#tr = transforms.Compose([
#    transforms.Resize(size=(224, 224)),
#
#    transforms.RandomRotation(degrees=15),
#    transforms.RandomVerticalFlip(p=0.5),
#    transforms.RandomHorizontalFlip(p=0.5),
#
#    transforms.RandomAdjustSharpness(sharpness_factor=5, p=.5),
#    transforms.RandomEqualize(p=.35),
#    transforms.ColorJitter(brightness=0.2, saturation=1.5, hue=.5, contrast=1.5),
#    transforms.RandomInvert(p=0.1),
#
#    transforms.ToTensor(),
#    transforms.RandomErasing(p=1., scale=(0.01, 0.01), ratio=(50, 80), value=0, inplace=False),
#    transforms.RandomErasing(p=1., scale=(0.01, 0.01), ratio=(1. / 80., 1. / 50.), value=0, inplace=False),


    # transforms.RandomRotation(degrees=15),
    #　transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomHorizontalFlip(p=0.5),

    # transforms.RandomAdjustSharpness(sharpness_factor=5, p=1.),
    # transforms.RandomEqualize(p=0),

    # transforms.ColorJitter(brightness=0.2, saturation=1.5, hue=.5, contrast=1.5),

    # transforms.RandomInvert(p=1.),
    # transforms.ToTensor(),
    # transforms.RandomErasing(p=.5, scale=(0.01, 0.01), ratio=(50, 80), value=0, inplace=False),
    # transforms.RandomErasing(p=.5, scale=(0.01, 0.01), ratio=(1. / 80., 1. / 50.), value=0, inplace=False),

])

img = tr(img)
# plt.imshow(img.numpy().transpose(1, 2, 0))
plt.imsave('kkkkk.jpg', img.numpy().transpose(1, 2, 0))
# plt.show()
