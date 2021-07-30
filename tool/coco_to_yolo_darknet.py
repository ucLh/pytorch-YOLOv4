# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/08 11:45
@Author        : Tianxiaomo
@File          : coco_annotatin.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import argparse
import json
from collections import defaultdict
from tqdm import tqdm
import os
import sys


def main(args):
    """load json file"""
    name_box_id = defaultdict(list)
    id_name = dict()
    with open(args.json_file_path, encoding='utf-8') as f:
        data = json.load(f)

    """generate labels"""
    images = data['images']
    annotations = data['annotations']
    real_names = os.listdir(args.images_dir_path)
    names_dict = {}

    for i, img in tqdm(enumerate(images)):
        id_ = img['id']
        name = img['file_name']
        w = img['width']
        h = img['height']
        names_dict[id_] = (name, w, h)

    for ant in tqdm(annotations):
        id = ant['image_id']
        # name = os.path.join(images_dir_path, images[id]['file_name'])
        # if not (names_dict[id] in real_names):
        #     continue
        name = os.path.join(args.images_dir_path, names_dict[id][0])
        cat = ant['category_id']

        if cat >= 1 and cat <= 11:
            cat = cat - 1
        elif cat >= 13 and cat <= 25:
            cat = cat - 2
        elif cat >= 27 and cat <= 28:
            cat = cat - 3
        elif cat >= 31 and cat <= 44:
            cat = cat - 5
        elif cat >= 46 and cat <= 65:
            cat = cat - 6
        elif cat == 67:
            cat = cat - 7
        elif cat == 70:
            cat = cat - 9
        elif cat >= 72 and cat <= 82:
            cat = cat - 10
        elif cat >= 84 and cat <= 90:
            cat = cat - 11

        name_box_id[name].append([ant['bbox'], cat, names_dict[id][1], names_dict[id][2]])

    used_names = dict()
    """write to txt"""
    with open(args.output_path, 'w') as f:
        for key in tqdm(name_box_id.keys()):
            f.write(key)
            f.write('\n')
            box_infos = name_box_id[key]
            # name = os.path.basename(key)[:-3] + 'txt'
            # if name in used_names.keys():
            #     print(key, used_names[name])
            #     print('!!!!!!!!!!!!!!')
            # used_names[name] = key
            save_path = key[:-3] + 'txt'
            with open(save_path, 'w') as f_w:
                for info in box_infos:
                    w_img = info[2]
                    h_img = info[3]
                    w = int(info[0][2])
                    h = int(info[0][3])
                    x_c = int(info[0][0]) + w / 2
                    y_c = int(info[0][1]) + h / 2

                    x_c /= w_img
                    w /= w_img
                    y_c /= h_img
                    h /= h_img
                    label = int(info[1])
                    if label == 0:
                        label = 1
                    elif label == 1:
                        label = 0

                    res_line = "%d %.3f %.3f %.3f %.3f\n" % (label, x_c, y_c, w, h)
                    f_w.write(res_line)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--json_file_path', type=str,
                        help='Path to file with json annotations',
                        default='../instances_train_cars_people.json')
    parser.add_argument('--images_dir_path', type=str,
                        help='Path to a csv file with annotations',
                        default='/home/integrant/Documents/ucLh/Programming/Python/Datasets/TRASH/train')
                        # default='/mnt/Luch/TRASH/')
    parser.add_argument('--output_path', type=str,
                        help='Path to save the resulting json annotations',
                        default='../data/train_pct.txt')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
