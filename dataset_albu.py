# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:09
@Author        : Tianxiaomo
@File          : dataset.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
from functools import partial
import os
import random
import sys

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset


class Yolo_dataset(Dataset):
    def __init__(self, lable_path, cfg, train=True):
        super(Yolo_dataset, self).__init__()
        if cfg.mixup == 2:
            print("cutmix=1 - isn't supported for Detector")
            raise
        elif cfg.mixup == 2 and cfg.letter_box:
            print("Combination: letter_box=1 & mosaic=1 - isn't supported, use only 1 of these parameters")
            raise

        self.cfg = cfg
        self.train = train
        if train:
            self.transforms = self.get_train_transforms()
        # else:
        #     self.transforms = self.get_valid_transforms()

        truth = {}
        f = open(lable_path, 'r', encoding='utf-8')
        for line in f.readlines():
            data = line.split(" ")
            truth[data[0]] = []
            for i in data[1:]:
                truth[data[0]].append([int(float(j)) for j in i.split(',')])

        self.truth = truth
        self.imgs = list(self.truth.keys())

    def __len__(self):
        return len(self.truth.keys())

    @staticmethod
    def get_train_transforms():
        return A.Compose(
            [
                # A.RandomSizedCrop(min_max_height=(380, 676), height=1024, width=1024, p=0.5),
                A.RandomSizedBBoxSafeCrop(608, 608, erosion_rate=0.0, interpolation=1, p=1),

                A.ToGray(p=0.01),
                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.Resize(height=608, width=608, p=1),
                A.Cutout(num_holes=4, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
                # A.Normalize(p=1),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
            )
        )

    # @staticmethod
    # def get_valid_transforms():
    #     return A.Compose(
    #         [
    #             A.Resize(height=608, width=608, p=1.0),
    #             A.Normalize(p=1),
    #             ToTensorV2(p=1.0),
    #         ],
    #         p=1.0,
    #         bbox_params=A.BboxParams(
    #             format='pascal_voc',
    #             min_area=0,
    #             min_visibility=0,
    #             label_fields=['labels']
    #         )
    #     )

    @staticmethod
    def load_image(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        return img

    def __getitem__(self, index):
        if not self.train:
            return self._get_val_item(index)
        img_path = self.imgs[index]
        # [x1, y1, x2, y2]
        boxes = np.array(self.truth.get(img_path), dtype=np.float)
        image = self.load_image(img_path)

        target = {'boxes': boxes, 'image_id': torch.tensor([index])}

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    tmp = tuple(map(lambda x: torch.tensor(x, dtype=torch.double), zip(*sample['bboxes'])))
                    target['boxes'] = torch.stack(tmp).permute(1, 0)
                    break
        out_bboxes = np.zeros([self.cfg.boxes, 5])
        out_bboxes[:min(target['boxes'].shape[0], self.cfg.boxes)] = \
            target['boxes'][:min(target['boxes'].shape[0], self.cfg.boxes)]
        return image, out_bboxes

    def _get_val_item(self, index):
        """
        """
        img_path = self.imgs[index]
        # print('val image path: '+ img_path)
        bboxes_with_cls_id = np.array(self.truth.get(img_path), dtype=np.float)
        img = cv2.imread(os.path.join(self.cfg.data_dir, img_path))
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # img = (img - mean) / std
        num_objs = len(bboxes_with_cls_id)
        target = {}
        # boxes to coco format
        boxes = bboxes_with_cls_id[...,:4]
        boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # box width, box height
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(bboxes_with_cls_id[...,-1].flatten(), dtype=torch.int64)
        target['image_id'] = torch.tensor([get_image_id(img_path)])
        target['area'] = (target['boxes'][:,3])*(target['boxes'][:,2])
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        return img, target


def get_image_id(filename:str) -> int:
    """
    Convert a string to a integer.
    Make sure that the images and the `image_id`s are in one-one correspondence.
    There are already `image_id`s in annotations of the COCO dataset,
    in which case this function is unnecessary.
    For creating one's own `get_image_id` function, one can refer to
    https://github.com/google/automl/blob/master/efficientdet/dataset/create_pascal_tfrecord.py#L86
    or refer to the following code (where the filenames are like 'level1_123.jpg')
    >>> lv, no = os.path.splitext(os.path.basename(filename))[0].split("_")
    >>> lv = lv.replace("level", "")
    >>> no = f"{int(no):04d}"
    >>> return int(lv+no)
    """
    no = os.path.splitext(os.path.basename(filename))[0].split("_")[-1]
    return int(no)
    # raise NotImplementedError("Create your own 'get_image_id' function")
    # return int(lv+no)


def draw_box(img, bboxes):
    for b in bboxes:
        img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    return img


if __name__ == "__main__":
    from cfg import Cfg
    import matplotlib.pyplot as plt

    random.seed(2020)
    np.random.seed(2020)
    Cfg.data_dir = '/mnt/e/Dataset'
    dataset = Yolo_dataset(Cfg.train_label, Cfg)
    for i in range(100):
        out_img, out_bboxes = dataset.__getitem__(i)
        a = draw_box(out_img.copy(), out_bboxes.astype(np.int32))
        plt.imshow(a.astype(np.int32))
        plt.show()
