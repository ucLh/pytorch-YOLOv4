import json
import os
import pandas as pd

from os import path as osp

json_dict = {
    "info": {
        "version": 1.0,
        "url": "no url specified",
        "year": 2021,
        "date_created": "today",
        "contributor": "no contributor specified",
        "description": ""
      },
    "categories": [
        {
          "name": "car",
          "id": 3,
          "supercategory": "unknown"
        }
    ],
    "licenses": [],
    "images": [],
    "annotations": []
}

image_dir = '/home/luch/Programming/Python/autovision/Yet-Another-EfficientDet-Pytorch/datasets' \
            '/detection_dataset/train'
image_names = os.listdir(image_dir)
image_names = sorted(image_names)
image_paths = list(map(lambda x: osp.join(image_dir, x), image_names))

df = pd.read_csv('/home/luch/Programming/Python/autovision/Yet-Another-EfficientDet-Pytorch/datasets'
                 '/detection_dataset/train_bbox_noisy.csv')

bbox_id = 0
for i, name in enumerate(image_names):
    image_base_entry = {
        "extra_info": {},
        "subdirs": ".",
        "id": i,
        "width": 676,
        "height": 380,
        "file_name": name,
    }
    json_dict["images"].append(image_base_entry)

    bboxes = df.loc[df['image'] == name]
    list_bboxes = bboxes.values.tolist()
    if not list_bboxes:
        continue

    for entry in list_bboxes:
        bbox_id += 1
        box = entry[1:]
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        # x_center, y_center, width, height = box
        # x_min = x_center - (width / 2)
        # y_min = y_center - (height / 2)
        new_box = [x_min, y_min, width, height]

        bbox_base_entry = {
          "image_id": i,
          "extra_info": {
            "human_annotated": True
          },
          "category_id": 3,
          "iscrowd": 0,
          "id": bbox_id,
          "bbox": new_box,
          "area": width * height
        }
        json_dict["annotations"].append(bbox_base_entry)

with open('train_annotations_noisy.json', 'w') as f:
    json.dump(json_dict, f, indent=4)

# df = pd.read_csv('/home/luch/Programming/Python/TestTasks/detection_dataset/train_bbox.csv')
# df.set_index('image', inplace=True)
# a = df.loc[df['image'] == 'vid_4_9580.jpg']
# lol = a.values.tolist()
# print(df['vid_4_9580.jpg'])
