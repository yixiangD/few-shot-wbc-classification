import os
import shutil

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator,\
        array_to_img, img_to_array, load_img

import hyper


def affine_transform(in_dir, minority_class='LYMPHOCYTE', num=2):
    if in_dir[-1] == '/':
        out_dir = '{}_aug/'.format(in_dir[:-1])
    else:
        out_dir = '{}_aug/'.format(in_dir)
        in_dir += '/'

    if os.path.exists(out_dir) and os.path.isdir(out_dir):
        print('Destination folder exist, cleaning files in destination folder')
        shutil.rmtree(out_dir)
    shutil.copytree(in_dir, out_dir)

    data_gen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=90,
        shear_range=0.1,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')
    for root, dirs, files in os.walk(in_dir + minority_class):
        classes = dirs
        files = files
        break

    for f in files:
        fname = "/".join([root, f])
        print(f"Processing {fname}")
        img = load_img(fname)
        img = img_to_array(img)
        img = img.reshape((1, ) + img.shape)
        i = 0
        for batch in data_gen.flow(x=img,
                                   save_to_dir=out_dir + minority_class,
                                   batch_size=16,
                                   shuffle=False,
                                   save_format="jpeg",
                                   save_prefix='image'):
            i += 1
            if i > num:
                break


if __name__ == '__main__':
    in_dir = '../data/split2/train/'
    affine_transform(in_dir)
