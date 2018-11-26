# -*- coding: utf-8 -*-
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import array_to_img
import os
import sys
import numpy as np
import random
from imgaug import augmenters as iaa
import imgaug as ia
import cv2
import pandas as pd
from keras.utils import Sequence
import keras
import math
from keras import backend as K
import logging

logging.basicConfig(level=logging.INFO,
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./log/stdout.log',
                    filemode='a')

with open('./data/resnet2/class_resnet2.txt') as f:
    filelist = [w.strip() for w in f.readlines()]

np.random.seed(1024)

# 按短边缩放,降低直接reshape造成的物体变形

def center_crop(x, center_crop_size):

    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    cropped = x[centerw - halfw: centerw + halfw,
              centerh - halfh: centerh + halfh, :]

    return cropped

def pad_to_bounding_box(image, target_height,target_width):
    # is_batch = True
    image_shape = image.shape

    height, width, depth = (image_shape[0], image_shape[1], image_shape[2])
    after_padding_width_left = (target_width - width)//2
    after_padding_width_right = after_padding_width_left + (target_width - width) % 2
    after_padding_height_top = (target_height - height)//2
    after_padding_height_bottom = after_padding_height_top + (target_height - height) % 2

    # Do not pad on the depth dimensions.
    paddings = np.reshape(
        np.stack([after_padding_height_top,after_padding_height_bottom,after_padding_width_left,
                         after_padding_width_right,0,0]), [3, 2])

    padded = np.pad(image, paddings, 'constant', constant_values=(70, 70))

    return padded

# 读取图片,并得到固定大小
def scale_byRatio(img_path, ratio=1.0, return_width=299, crop_method=center_crop):
    # Given an image path, return a scaled array

    try:
        img_path_gbk = img_path.encode("GB18030")
        img = cv2.imread(img_path_gbk.decode())
    except:
        try:
            img_path_gbk = img_path.encode("UTF-8")
            img = cv2.imread(img_path_gbk.decode())
        except:
            logging.error("read image error: %s %s" % (sys.exc_info(), img_path))
            return None

    try:
        if img is not None:
            h, w, _ = img.shape

            # shorter = min(w, h)
            # img_cropped = crop_method(img, (shorter, shorter))
            img_resized = cv2.resize(img, (0, 0), fx=return_width / w, fy=return_width / h, interpolation=cv2.INTER_CUBIC)

            # longer = max(w, h)
            # if longer > return_width:
            #     img = cv2.resize(img, (0, 0), fx=return_width / longer, fy=return_width / longer,
            #                      interpolation=cv2.INTER_CUBIC)
            # img_resized = pad_to_bounding_box(img, return_width, return_width)

            img_rgb = img_resized
            # img_rgb[:, :, [0, 1, 2]] = img_resized[:, :, [2, 1, 0]]
            return img_rgb
        else:
            logging.info("can not read image in img_path= %s" % img_path)
            return None
    except:
        logging.error("image np error: %s %s" % (sys.exc_info(), img_path))
        return None



# 数据扩增库imgaug的应用
"""
# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
st = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        #iaa.Flipud(0.5), # vertically flip 50% of all images
        st(iaa.Crop(percent=(0, 0.15))), # crop images by 0-15% of their height/width
        #st(iaa.GaussianBlur((0, 2.0))), # blur images with a sigma between 0 and 3.0
        st(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
        st(iaa.Multiply((0.85, 1.15), per_channel=0.5)), # change brightness of images (75-125% of original value)
        st(iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5)), # improve or worsen the contrast
        st(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
            #translate_px={"x": (-10, 10), "y": (-10, 10)}, # translate by -16 to +16 pixels (per axis)
            rotate=(-15, 15), # rotate by -10 to +10 degrees
            #shear=(-5, 5), # shear by -16 to +16 degrees
            order=ia.ALL, # use any of scikit-image's interpolation methods
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ))
    ],
    random_order=True # do all of the above in random order
)
"""

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.65, 1.15), "y": (0.65, 1.15)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, # translate by -20 to +20 percent (per axis)
            rotate=(-30, 30), # rotate by -45 to +45 degrees
            shear=(-5, 5), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 3),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(1, 5)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(1, 5)), # blur image using local medians with kernel sizes between 2 and 7
                ]),

                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.01, 0.03), per_channel=0.2),
                ]),
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)

                iaa.ContrastNormalization((0.3, 1.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
            ],
            random_order=True
        )
    ],
    random_order=True
)
# 伪迭代器,因使用了Sequence,可不用 yeild 改用return 返回
def generator_batch(data_list, nbr_classes=3, batch_size=32,batch_index = 0,return_label=True,
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, save_network_input=None, augment=False):
    '''
    A generator that yields a batch of (data, label).

    Input:
        data_list  : a MxNet styple of data list, e.g.
                     "/data/workspace/dataset/Cervical_Cancer/train/Type_1/0.jpg 0"
        shuffle    : whether shuffle rows in the data_llist
        batch_size : batch size

    Output:
        (X_batch, Y_batch)
    '''

    N = len(data_list)

    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            #batch_index += 1
        else:
            current_batch_size = N - current_index
            #batch_index = 0


        X_batch = np.zeros((current_batch_size, img_width, img_height, 3))
        Y_batch = np.zeros((current_batch_size, nbr_classes))
        Y_batch = pd.DataFrame(Y_batch, columns=filelist)

        for i in range(current_index, current_index + current_batch_size):
            line = data_list[i].strip().split(' ')
            # print("%s line is: %s" % (batch_index, line))

            if return_label:
                label = str(line[-1])

            if len(line) == 2:
                img_path = line[0]
            else:
                img_path = ''.join(line[:-1])

            if random_scale:
                scale_ratio = random.uniform(0.9, 1.1)

            img = scale_byRatio(img_path, ratio=scale_ratio, return_width=img_width,
                                crop_method=crop_method)
            if img is None:
                continue

            X_batch[i - current_index,:,:,:] = img
            if return_label:
                Y_batch.ix[[i - current_index],[label]] = 1.0

        if augment:
            X_batch = X_batch.astype(np.float32)
            X_batch = seq.augment_images(X_batch)
            mixup = False
            # mixup 数据扩增,不懂请自行百度
            if mixup and current_batch_size > 10:
                alpha = 0.4
                lam = np.random.beta(alpha, alpha)
                for _ in range(5):
                    a, b, c = np.random.randint(0, current_batch_size, size=(1, 3))[0]
                    x1, x2, y1, y2 = X_batch[b,:,:,:], X_batch[c,:,:,:], Y_batch.ix[b], Y_batch.ix[c]
                    X_batch[a,:,:,:] = x1 * lam + x2 * (1-lam)
                    Y_batch.ix[a] = y1 * lam + y2 * (1 - lam)

                del a, b, c, x1, x2 ,y1 ,y2



        if save_to_dir:
            for i in range(current_index, current_index + current_batch_size):
                tmp_path = data_list[i].strip().split(' ')[0]
                basedir = tmp_path.split(os.sep)[-2:]
                image_name = '_'.join(basedir)
                img_to_save_path = os.path.join(save_to_dir, image_name)
                img = array_to_img(X_batch[i - current_index])
                img.save(img_to_save_path)

        X_batch = X_batch.astype(np.float32)
        X_batch = preprocess_input(X_batch)


        if save_network_input:
            print('X_batch.shape: {}'.format(X_batch.shape))
            X_to_save = X_batch.reshape((299, 299, 3))
            to_save_base_name = save_network_input[:-4]
            np.savetxt(to_save_base_name + '_0.txt', X_to_save[:, :, 0], delimiter=' ')
            np.savetxt(to_save_base_name + '_1.txt', X_to_save[:, :, 1], delimiter=' ')
            np.savetxt(to_save_base_name + '_2.txt', X_to_save[:, :, 2], delimiter=' ')

        # img = X_batch[0,:,:,:]
        # img = np.reshape(img, (-1))
        if return_label:
            return (X_batch, Y_batch)
        else:
            return X_batch


# 保存多GPU权重
class CustomModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, model, path, monitor_index):
        self.model = model
        self.path = path
        self.monitor_index = monitor_index
        self.best_acc = 0.90

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs[self.monitor_index]
        if val_acc > self.best_acc:
            print("\nValidation accuracy increased from {} to {}, saving model".format(self.best_acc, val_acc))
            self.model.save_weights(self.path, overwrite=True)
            self.best_acc = val_acc
            
            
# 正式迭代器版本
class SequenceData(Sequence):
    def __init__(
                self, data_list, nbr_classes=3, batch_size=32, return_label=True,
                scale_ratio=1.0, random_scale=False,
                img_width=299, img_height=299, shuffle=True,
                save_to_dir=None, save_network_input=None, augment=False
                ):
        # 初始化所需的参数
        self.data_list = data_list
        self.nbr_classes = nbr_classes
        self.batch_size = batch_size
        self.return_label = return_label
        self.scale_ratio = scale_ratio
        self.random_scale = random_scale
        self.img_width = img_width
        self.img_height = img_height
        self.shuffle = shuffle
        self.save_to_dir = save_to_dir
        self.save_network_input = save_network_input
        self.augment = augment
        if self.shuffle:
            random.shuffle(data_list)

    def __len__(self):
        # 让代码知道这个序列的长度

        return math.ceil(len(self.data_list)/self.batch_size)

    def __getitem__(self, idx):
        return generator_batch(data_list=self.data_list, nbr_classes=self.nbr_classes, batch_size=self.batch_size,
                               batch_index=idx, return_label=True, scale_ratio=self.scale_ratio, random_scale=self.random_scale,
                               img_width=self.img_width, img_height=self.img_width, shuffle=self.shuffle,
                               save_to_dir=self.save_to_dir, save_network_input=self.save_network_input, augment=self.augment)


class SGDRScheduler(keras.callbacks.Callback):
    '''Schedule learning rates with restarts
     A simple restart technique for stochastic gradient descent.
    The learning rate decays after each batch and peridically resets to its
    initial value. Optionally, the learning rate is additionally reduced by a
    fixed factor at a predifined set of epochs.
     # Arguments
        epochsize: Number of samples per epoch during training.
        batchsize: Number of samples per batch during training.
        start_epoch: First epoch where decay is applied.
        epochs_to_restart: Initial number of epochs before restarts.
        mult_factor: Increase of epochs_to_restart after each restart.
        lr_fac: Decrease of learning rate at epochs given in
                lr_reduction_epochs.
        lr_reduction_epochs: Fixed list of epochs at which to reduce
                             learning rate.
     # References
        - [SGDR: Stochastic Gradient Descent with Restarts](http://arxiv.org/abs/1608.03983)
    '''

    def __init__(self,
                 epochsize,
                 batchsize,
                 epochs_to_restart=2,
                 mult_factor=2,
                 lr_fac=0.1,
                 lr_reduction_epochs=(40, 80, 120)):
        super(SGDRScheduler, self).__init__()
        self.epoch = -1
        self.batch_since_restart = 0
        self.next_restart = epochs_to_restart
        self.epochsize = epochsize
        self.batchsize = batchsize
        self.epochs_to_restart = epochs_to_restart
        self.mult_factor = mult_factor
        self.batches_per_epoch = self.epochsize / self.batchsize
        self.lr_fac = lr_fac
        self.lr_reduction_epochs = lr_reduction_epochs
        self.lr_log = []

    def on_train_begin(self, logs={}):
        self.lr = K.get_value(self.model.optimizer.lr)

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1

    def on_batch_end(self, batch, logs={}):
        fraction_to_restart = self.batch_since_restart / \
            (self.batches_per_epoch * self.epochs_to_restart)
        lr = 0.5 * self.lr * (1 + np.cos(fraction_to_restart * np.pi))
        K.set_value(self.model.optimizer.lr, lr)
        self.batch_since_restart += 1
        self.lr_log.append(lr)

    def on_epoch_end(self, epoch, logs={}):
        if self.epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.epochs_to_restart *= self.mult_factor
            self.next_restart += self.epochs_to_restart

        if (self.epoch + 1) in self.lr_reduction_epochs:
            self.lr *= self.lr_fac
