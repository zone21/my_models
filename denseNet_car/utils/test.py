import cv2
import sys
import numpy as np
import numpy as array_ops
# from keras.applications.imagenet_utils import preprocess_input
from imgaug import augmenters as iaa
import imgaug as ia
import math

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


def pad_to_bounding_box(image, target_height,
                        target_width):
    # is_batch = True
    image_shape = image.shape
    # if array_ops.ndim(image) == 3:
    #     is_batch = False
    #     image = array_ops.expand_dims(image, 0)
    # elif array_ops.ndim(image) is None:
    #     is_batch = False
    #     image = array_ops.expand_dims(image, 0)
    #     image.set_shape([None] * 4)
    # elif array_ops.ndim(image) != 4:
    #     raise ValueError('\'image\' must have either 3 or 4 dimensions.')

    height, width, depth = (image_shape[0], image_shape[1], image_shape[2])
    after_padding_width_left = target_width//2 - width//2 + (target_width - width) % 2
    after_padding_width_right = target_width//2 - width//2
    after_padding_height_top = target_height//2 - height//2 + (target_height - height) % 2
    after_padding_height_bottom = target_height//2 - height//2

    # Do not pad on the depth dimensions.
    paddings = array_ops.reshape(
        array_ops.stack([after_padding_height_top,after_padding_height_bottom,after_padding_width_left,after_padding_width_right,0,0]), [3, 2])

    padded = array_ops.pad(image, paddings, 'constant', constant_values=(70,70))

    # padded_shape = [
    #     i for i in [target_height, target_width, depth]
    #     ]
    # padded.reshape(padded_shape)

    return padded

def center_crop(x, center_crop_size):

    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    cropped = x[centerw - halfw: centerw + halfw,
              centerh - halfh: centerh + halfh, :]

    return cropped

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
            print("read image error: %s %s" % (sys.exc_info(), img_path))
            return None

    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w, _ = img.shape
        longer = max(w, h)
        # img_cropped = crop_method(img, (shorter, shorter))
        if longer > return_width:
            img = cv2.resize(img, (0, 0), fx=return_width / longer, fy=return_width / longer, interpolation=cv2.INTER_CUBIC)
        img_resized = pad_to_bounding_box(img, return_width, return_width)
        img_rgb = img_resized
        # img_rgb[:, :, [0, 1, 2]] = img_resized[:, :, [2, 1, 0]]
        return img_rgb
    else:
        print("can not read image in img_path= %s" % img_path)
        return None


if __name__ == '__main__':
    scale_ratio = 1.0
    crop_method = center_crop
    X_batch = np.zeros((1, 299, 299, 3))

    img = scale_byRatio('./d2.png', ratio=scale_ratio, return_width=197, crop_method=crop_method)
    #
    # X_batch = X_batch.astype(np.float32)
    #
    # p = np.array(img, dtype=np.float32)
    # p = seq.augment_images(p)
    # # p = np.expand_dims(p, axis=0)
    # p = np.array(p, dtype=np.float32)
    # p = preprocess_input(p)
    #
    # # image = cv2.imdecode(p, 0)
    #
    cv2.imwrite("test5.jpg", img)