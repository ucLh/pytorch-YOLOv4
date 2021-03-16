import csv
import os
from os import path as osp

annot = '/home/luch/Programming/Python/TestTasks/detection_dataset/train_bbox.csv'
image_dir = '/home/luch/Programming/Python/TestTasks/detection_dataset/train'
known_names = set()
all_train_images = os.listdir(image_dir)
all_image_names = sorted(all_train_images)
all_image_paths = list(map(lambda x: osp.join(image_dir, x), all_image_names))
no_bbox_images = set()

with open(annot) as fp, open('output.txt', 'w') as out:
    reader = csv.reader(fp, delimiter=",")
    next(reader, None)  # skip the headers
    start_str = next(reader)
    cur_path = osp.abspath(osp.join(image_dir, start_str[0]))
    known_names.add(cur_path)
    box_str = f' {start_str[1]},{start_str[2]},{start_str[3]},{start_str[4]},2'
    result = cur_path + box_str
    for row in reader:
        new_path = osp.abspath(osp.join(image_dir, row[0]))
        known_names.add(new_path)
        box_str = f' {row[1]},{row[2]},{row[3]},{row[4]},2'
        if new_path != cur_path:
            out.write(result + '\n')
            result = new_path + box_str
        else:
            result += box_str
        cur_path = new_path
    out.write(result + '\n')
    for path in all_image_paths:
        if path not in known_names:
            out.write(path + '\n')
            # no_bbox_images.add(path)

    # for path in list(no_bbox_images)[70:]:
    #     out.write(path + '\n')
