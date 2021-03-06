# !/usr/bin/env python2

from __future__ import print_function

import argparse
from random import randint

import numpy as np
from skimage.io import imread
from skimage.color import *
from skimage.transform import resize
from sklearn.model_selection import train_test_split

image_rows = 80
image_cols = 80

image_rows_rez = 80
image_cols_rez = 80

test_percentage = 0.15

num_classes = 2


def create_train_and_test_data(csv_path):

    ids = []
    imgs_gt = []

    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)

    dictGT = {}
    dictLabel = {}

    dictLabel['neutral'] = 0
    dictLabel['noneu'] = 1
    #dictLabel['neutral'] = 2
    #dictLabel['refill'] = 3

    images = []
    with open(csv_path, 'rb') as features:
        train = features.readlines()
        for i, line in enumerate(train):
            if (i != 0):
                f_info = line.decode().split(',')
                if not 'skip' in f_info[-1]:
                    dictGT[f_info[-2].split('/')[-1].split('.')[0]] = \
                        dictLabel[f_info[-1].replace('\n', '').replace('\r', '').replace('"', '')]
                    images.append(f_info[-2])
    features.close()

    total = len(images)
    channels = 3
    imgs_8bit = np.ndarray((total, image_rows_rez, image_cols_rez, channels), dtype=np.uint8)

    for idx, image_name in enumerate(images):
        im = imread(image_name)
        im = gray2rgb(im)
        img = resize(im, (image_rows_rez, image_cols_rez), preserve_range=True, mode='constant')

        imgs_8bit[idx] = np.array([img])
        ids.append(image_name.split('.')[0])
        gt = np.zeros(num_classes)
        gt[dictGT[image_name.split('/')[-1].split('.')[0]]] += 1
        imgs_gt.append(gt)

        if idx % 100 == 0:
            print('Done: {0}/{1} images'.format(idx, total))
    print('Loading done.')

    ids_array = np.array(ids, dtype=object)
    imgs_gt_array = np.array(imgs_gt, dtype=object)
    image_position = np.arange(total)

    image_position_train, image_position_test = \
        train_test_split(image_position, test_size=test_percentage, random_state=randint(0, 100))

    ids_train = create_subarray(image_position_train, ids_array)
    ids_test = create_subarray(image_position_test, ids_array)

    imgs_train_8bit = create_subarray(image_position_train, imgs_8bit)
    imgs_test_8bit = create_subarray(image_position_test, imgs_8bit)

    imgs_gt_train = create_subarray(image_position_train, imgs_gt_array)
    imgs_gt_test = create_subarray(image_position_test, imgs_gt_array)

    np.save('imgs_train_8bit.npy', imgs_train_8bit)
    np.save('imgs_test_8bit.npy', imgs_test_8bit)
    np.save('imgs_train_gt.npy', imgs_gt_train)
    np.save('imgs_test_gt.npy', imgs_gt_test)
    np.save('ids_train.npy', ids_train)
    np.save('ids_test.npy', ids_test)
    print('Saving to .npy files done.')


def create_subarray(subset_index, total_array):
    result_array = []
    for value in subset_index:
        result_array.append(total_array[value])

    return result_array


def load_train_data():
    imgs_train = np.load('imgs_train_8bit.npy')
    imgs_train_gt = np.load('imgs_train_gt.npy')
    imgs_train_id = np.load('ids_train.npy')

    return imgs_train, imgs_train_gt, imgs_train_id


def load_test_data():
    imgs_test = np.load('imgs_test_8bit.npy')
    imgs_test_gt = np.load('imgs_test_gt.npy')
    imgs_test_id = np.load('ids_test.npy')

    return imgs_test, imgs_test_gt, imgs_test_id


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="")
    p.add_argument('-data', dest='data', action='store', default='data.csv', help='data path file *.csv')

    args = p.parse_args()

    create_train_and_test_data(args.data)


if __name__ == '__main__':
    main()
