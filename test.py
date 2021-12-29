import os
import config_infer as config

from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
import time
import csv
import logging
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import WSDAN_CAL
from datasets import get_fish_test_dataset
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment

# GPU settings
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# General loss functions
cross_entropy_loss = nn.CrossEntropyLoss()
center_loss = CenterLoss()

# loss and metric
loss_container = AverageMeter(name='loss')
raw_metric = TopKAccuracyMetric(topk=(1, 5))
crop_metric = TopKAccuracyMetric(topk=(1, 5))
drop_metric = TopKAccuracyMetric(topk=(1, 5))

best_acc = 0.0
from torchvision import transforms

ToPILImage = transforms.ToPILImage()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)


def main():
    # Load dataset
    ##################################
    test_dataset = get_fish_test_dataset(config.image_size)

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.workers, pin_memory=True)
    num_classes = 8

    ##################################
    # Initialize model
    ##################################
    logs = {}
    start_epoch = 0
    net = WSDAN_CAL(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=True)

    checkpoint = torch.load(config.ckpt)
    state_dict = checkpoint['state_dict']
    net.load_state_dict(state_dict)
    print('Network loaded from {}'.format(config.ckpt))

    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    test(data_loader=test_loader, net=net)


def test(**kwargs):
    # Retrieve training configuration
    global best_acc
    data_loader = kwargs['data_loader']
    net = kwargs['net']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    drop_metric.reset()

    # begin validation
    start_time = time.time()
    net.eval()
    softmax = nn.Softmax(dim=1)
    all_pred = []
    all_names = []

    all_pred_top1 = []
    all_pred_top3 = []
    with torch.no_grad():
        for i, (X, names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # obtain data
            X = X.to(device)

            X_m = torch.flip(X, [3])

            ##################################
            # Raw Image
            ##################################
            y_pred_raw, y_pred_aux_raw, _, attention_map = net(X)
            y_pred_raw_m, y_pred_aux_raw_m, _, attention_map_m = net(X_m)

            ##################################
            # Object Localization and Refinement
            ##################################

            crop_images = batch_augment(X, attention_map, mode='crop', theta=0.3, padding_ratio=0.1)
            y_pred_crop, y_pred_aux_crop, _, _ = net(crop_images)

            crop_images2 = batch_augment(X, attention_map, mode='crop', theta=0.2, padding_ratio=0.1)
            y_pred_crop2, y_pred_aux_crop2, _, _ = net(crop_images2)

            crop_images3 = batch_augment(X, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop3, y_pred_aux_crop3, _, _ = net(crop_images3)

            crop_images_m = batch_augment(X_m, attention_map_m, mode='crop', theta=0.3, padding_ratio=0.1)
            y_pred_crop_m, y_pred_aux_crop_m, _, _ = net(crop_images_m)

            crop_images_m2 = batch_augment(X_m, attention_map_m, mode='crop', theta=0.2, padding_ratio=0.1)
            y_pred_crop_m2, y_pred_aux_crop_m2, _, _ = net(crop_images_m2)

            crop_images_m3 = batch_augment(X_m, attention_map_m, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop_m3, y_pred_aux_crop_m3, _, _ = net(crop_images_m3)

            y_pred = (y_pred_raw + y_pred_crop + y_pred_crop2 + y_pred_crop3) / 4.
            y_pred_m = (y_pred_raw_m + y_pred_crop_m + y_pred_crop_m2 + y_pred_crop_m3) / 4.
            y_pred = (y_pred + y_pred_m) / 2.

            y_pred_aux = (y_pred_aux_raw + y_pred_aux_crop + y_pred_aux_crop2 + y_pred_aux_crop3) / 4.
            y_pred_aux_m = (y_pred_aux_raw_m + y_pred_aux_crop_m + y_pred_aux_crop_m2 + y_pred_aux_crop_m3) / 4.
            y_pred_aux = (y_pred_aux + y_pred_aux_m) / 2.

            pred = softmax(y_pred)
            new_pred_top1 = []
            new_pred_top3 = []
            for ii in range(len(pred)):
                one_pred = pred[ii]
                pred_with_idx = []
                for cls_idx, cls_p in enumerate(one_pred):
                    pred_with_idx.append([cls_p, cls_idx])
                pred_with_idx = sorted(pred_with_idx, reverse=True, key=lambda x: x[0])

                # Make new one_pred
                #     No.1~3 => 0.25
                #     No.4~8 => 0.05
                new_one_pred_top1 = [0., 0., 0., 0., 0., 0., 0., 0.]
                new_one_pred_top3 = [0., 0., 0., 0., 0., 0., 0., 0.]

                new_one_pred_top3[pred_with_idx[0][1]] = 0.25
                new_one_pred_top3[pred_with_idx[1][1]] = 0.25
                new_one_pred_top3[pred_with_idx[2][1]] = 0.25
                new_one_pred_top3[pred_with_idx[3][1]] = 0.05
                new_one_pred_top3[pred_with_idx[4][1]] = 0.05
                new_one_pred_top3[pred_with_idx[5][1]] = 0.05
                new_one_pred_top3[pred_with_idx[6][1]] = 0.05
                new_one_pred_top3[pred_with_idx[7][1]] = 0.05

                new_one_pred_top1[pred_with_idx[0][1]] = 0.507
                new_one_pred_top1[pred_with_idx[1][1]] = 0.07242857142
                new_one_pred_top1[pred_with_idx[2][1]] = 0.07242857142
                new_one_pred_top1[pred_with_idx[3][1]] = 0.07242857142
                new_one_pred_top1[pred_with_idx[4][1]] = 0.07242857142
                new_one_pred_top1[pred_with_idx[5][1]] = 0.07242857142
                new_one_pred_top1[pred_with_idx[6][1]] = 0.07242857142
                new_one_pred_top1[pred_with_idx[7][1]] = 0.07242857142

                new_pred_top3.append(new_one_pred_top3)
                new_pred_top1.append(new_one_pred_top1)

            pred_top3 = torch.tensor(new_pred_top3)
            pred_top1 = torch.tensor(new_pred_top1)

            all_pred += pred.tolist()
            all_names += list(names)
            all_pred_top1 += pred_top1.tolist()
            all_pred_top3 += pred_top3.tolist()

    os.makedirs('./pred', exist_ok=True)
    # Raw pred
    with open(os.path.join('./pred', "{}.csv".format(datetime.now().strftime("UTC+8_%Y_%m-%d_%H:%M"))),
              "w+") as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
        for i, pred in enumerate(all_pred):
            writer.writerow([os.path.relpath(all_names[i], '.')] + pred)
    print("Original pred done!")
    # Top-1 pred
    with open(os.path.join('./pred', "{}_top1.csv".format(datetime.now().strftime("UTC+8_%Y_%m-%d_%H:%M"))),
              "w+") as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
        for i, pred in enumerate(all_pred_top1):
            writer.writerow([os.path.relpath(all_names[i], '.')] + pred)
    print("Top-1 pred done!")
    # Top-3 pred
    with open(os.path.join('./pred', "{}_top3.csv".format(datetime.now().strftime("UTC+8_%Y_%m-%d_%H:%M"))),
              "w+") as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
        for i, pred in enumerate(all_pred_top3):
            writer.writerow([os.path.relpath(all_names[i], '.')] + pred)
    print("Top-3 pred done!")
    #


if __name__ == '__main__':
    main()
