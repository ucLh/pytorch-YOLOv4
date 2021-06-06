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
from PIL import Image
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

        truth = {}
        f = open(lable_path, 'r', encoding='utf-8')
        for line in f.readlines():
            data = line.split(';')
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


    @staticmethod
    def load_image(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        return img

    @staticmethod
    def load_image_pil(path):
        img = Image.open(path)
        img = np.array(img, dtype=np.float32)
        img = img / 255
        return img

    def __getitem__(self, index):
        if not self.train:
            return self._get_val_item_offline_wgisd(index)
        img_path = self.imgs[index]
        # [x1, y1, x2, y2]
        boxes = np.array(self.truth.get(img_path), dtype=np.float)
        image = self.load_image_pil(img_path)

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

    def _get_val_item_offline(self, index):
        img_path = self.imgs[index]
        # print('val image path: '+ img_path)
        bboxes_with_cls_id = np.array(self.truth.get(img_path), dtype=np.float)

        img_old = cv2.imread(os.path.join(self.cfg.data_dir, img_path))
        img = self.load_image_pil(os.path.join(self.cfg.data_dir, img_path))
        w, h, h_cut = 2432, 1368, 1216

        # 0. Dict for all augmentations
        bbox_params_ = {
            'format': 'pascal_voc',
            'min_area': 0,
            'min_visibility': 0,
        }

        # 1. Resize the image, and crop out upper part of it
        resize_and_crop = A.Compose([
            A.Resize(height=h, width=w, p=1.0),
            A.Crop(x_min=0, x_max=w, y_min=h - h_cut, y_max=1368, p=1.0)
        ],
            p=1.0,
            bbox_params=A.BboxParams(**bbox_params_)
        )

        # 2. Divide into 8 squares
        assert (w % 4) == 0
        assert (h % 2) == 0
        sides_x = [int(w * (j / 4)) for j in range(5)]
        sides_y = [int(h_cut * (j / 2)) for j in range(3)]

        crop_list = []
        for i in range(4):
            for j in range(2):
                x_min, x_max = sides_x[i], sides_x[i+1]
                y_min, y_max = sides_y[j], sides_y[j+1]
                crop_list.append(
                    A.Compose([
                        A.Crop(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, p=1.0),
                        # ToTensorV2(p=1.0)
                    ],
                        p=1.0,
                        bbox_params=A.BboxParams(**bbox_params_))
                )

        # 3. Apply augs
        temp = resize_and_crop(**{
                    'image': img,
                    'bboxes': bboxes_with_cls_id,
                })
        images_list, boxes_list = [], []
        for crop in crop_list:
            sample = crop(**{
                    'image': temp['image'],
                    'bboxes': temp['bboxes'],
                })
            images_list.append(sample['image'])
            boxes_list.append(sample['bboxes'])
        # images_combined = torch.stack(images_list, dim=0)
        for i in range(len(boxes_list)):
            boxes_list[i] = tuple(map(lambda x: torch.tensor(x, dtype=torch.float), boxes_list[i]))
        boxes_combined = pad_annots(boxes_list)

        targets_list = []
        for box in boxes_combined:
            num_objs = len(box)
            target = {}
            # boxes to coco format
            boxes = box[..., :4]
            boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # box width, box height
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(bboxes_with_cls_id[...,-1].flatten(), dtype=torch.int64)
            # target['image_id'] = torch.tensor([get_image_id(img_path)])
            target['area'] = (target['boxes'][:,3])*(target['boxes'][:,2])
            target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
            targets_list.append(target)

        return images_list, targets_list

    def _get_val_item_offline_wgisd(self, index):
        img_path = self.imgs[index]
        # print('val image path: '+ img_path)
        bboxes_with_cls_id = np.array(self.truth.get(img_path), dtype=np.float)

        img = Image.open(img_path)
        img = np.array(img)
        w, h, h_cut = 4256, 2432, 2432

        # 0. Dict for all augmentations
        bbox_params_ = {
            'format': 'pascal_voc',
            'min_area': 0,
            'min_visibility': 0,
        }

        # 1. Resize the image, and crop out upper part of it
        resize_and_crop = A.Compose([
            A.Resize(height=h, width=w, p=1.0),
            # A.Crop(x_min=0, x_max=w, y_min=h - h_cut, y_max=h, p=1.0)
        ],
            p=1.0,
            bbox_params=A.BboxParams(**bbox_params_)
        )

        # 2. Divide into 8 squares
        assert (w % 7) == 0
        assert (h % 4) == 0
        sides_x = [int(w * (j / 7)) for j in range(8)]
        sides_y = [int(h_cut * (j / 4)) for j in range(5)]

        crop_list = []
        for i in range(7):
            for j in range(4):
                x_min, x_max = sides_x[i], sides_x[i+1]
                y_min, y_max = sides_y[j], sides_y[j+1]
                crop_list.append(
                    A.Compose([
                        A.Crop(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, p=1.0),
                        # ToTensorV2(p=1.0)
                    ],
                        p=1.0,
                        bbox_params=A.BboxParams(**bbox_params_))
                )

        # 3. Apply augs
        try:
            temp = resize_and_crop(**{
                        'image': img,
                        'bboxes': bboxes_with_cls_id,
                    })
        except:
            return None, None
        images_list, boxes_list = [], []
        for crop in crop_list:
            sample = crop(**{
                    'image': temp['image'],
                    'bboxes': temp['bboxes'],
                })
            images_list.append(sample['image'])
            boxes_list.append(sample['bboxes'])
        # images_combined = torch.stack(images_list, dim=0)
        for i in range(len(boxes_list)):
            boxes_list[i] = tuple(map(lambda x: torch.tensor(x, dtype=torch.float), boxes_list[i]))
        # boxes_combined = pad_annots(boxes_list)

        targets_list = []
        for box in boxes_list:
            num_objs = len(box)
            target = {}
            # boxes to coco format

            target['num_objs'] = num_objs
            targets_list.append(target)

        return images_list, targets_list


def pad_annots(annots):
    max_num_annots = max(len(annot) for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if len(annot) > 0:
                annot = torch.stack(annot, dim=0)
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1
    return annot_padded


if __name__ == "__main__":
    import sys
    sys.path.append('../')
    from cfg import Cfg
    from tqdm import tqdm
    from pathlib import Path
    import json

    random.seed(2020)
    np.random.seed(2020)
    Cfg.mixup = 0
    Cfg.data_dir = '/home/luch/Programming/Python/autovision/wgisd/box_val'
    dataset = Yolo_dataset('data/test_berries.txt', Cfg, train=False)
    json_base = {
        "licenses": [
            {
                "name": "",
                "id": 0,
                "url": ""
            }
        ],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "",
            "url": "",
            "version": "",
            "year": ""
        },
        "categories": [
            {
                "id": 1,
                "name": "Berry",
                "supercategory": ""
            },
        ],
        "images": [],
        "annotations": []
    }

    image_counter, annot_counter = 1, 1
    for i in tqdm(range(len(dataset))):
        out_imgs, out_bboxes_s = dataset.__getitem__(i)
        for j, img in enumerate(out_imgs):
            img_pil = Image.fromarray(img)
            boxes = out_bboxes_s[j]['boxes']
            valid_boxes_flag = False
            for box in boxes:
                if box[2] > 5 and box[3] > 5:
                    valid_boxes_flag = True
            if boxes[0][0] == -1 or (not valid_boxes_flag):
                continue
            img_pil.save(f'/home/luch/Programming/Python/autovision/wgisd/test_frac/{image_counter}.jpg')
            image_base_entry = {
                "extra_info": {},
                "subdirs": ".",
                "id": image_counter,
                "width": 608,
                "height": 608,
                "file_name": f'{image_counter}.jpg',
            }
            json_base['images'].append(image_base_entry)
            for box in boxes:
                box = box.numpy().tolist()
                if box[0] == -1 or box[2] <= 5 or box[1] <= 5:
                    continue

                bbox_base_entry = {
                    "image_id": image_counter,
                    "extra_info": {
                        "human_annotated": True
                    },
                    "category_id": 1,
                    "iscrowd": 0,
                    "id": annot_counter,
                    "bbox": box,
                    "area": box[2] * box[3]
                }
                annot_counter += 1
                json_base['annotations'].append(bbox_base_entry)
                box = list(map(int, box))
                # print(box)
                # img = cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 255, 255))
            # cv2.imwrite(f'/mnt/Luch/TRASH/test_frac/{image_counter}.jpg', img)

            image_counter += 1
    with open('/home/luch/Programming/Python/autovision/wgisd/test_frac/instances_test_fractured.json', 'w') as f:
        json.dump(json_base, f, indent=4)
