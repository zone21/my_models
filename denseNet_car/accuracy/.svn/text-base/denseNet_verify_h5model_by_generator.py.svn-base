#coding=GB18030
import argparse

import sys

import cv2
import os
import numpy as np
from keras import Model
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import model_from_json
from keras.layers import Dense, Dropout
from keras.applications.densenet import DenseNet201
from keras.optimizers import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.visible_device_list = '0,1'
config.gpu_options.per_process_gpu_memory_fraction = 0.04
set_session(tf.Session(config=config))

img_width = 299
img_height = 200
batch_size = 3000

def center_crop(x, center_crop_size):
    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    cropped = x[centerw - halfw: centerw + halfw,
              centerh - halfh: centerh + halfh, :]

    return cropped


# 读取图片,并得到固定大小
def scale_byRatio(img_path, ratio=1.0, return_width=200, crop_method=center_crop):
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

    h, w, _ = img.shape
    shorter = min(w, h)

    img_cropped = crop_method(img, (shorter, shorter))
    img_resized = cv2.resize(img_cropped, (return_width, return_width), interpolation=cv2.INTER_CUBIC)
    img_rgb = img_resized
    img_rgb[:, :, [0, 1, 2]] = img_resized[:, :, [2, 1, 0]]

    return img_rgb


def prepareData(args):
    data = {}
    classes_test_data = args.classes_test_data
    test_data_dir = args.test_data_dir
    serial_image_dir = {}
    if not os.path.exists(test_data_dir):
        print("image dir not exist........")
        return -1

    if classes_test_data == '':
        # 没有指定车系就跑image_dir下全部的目录
        for maindir, subdir, file_name_list in os.walk(test_data_dir):
            for dir in subdir:
                serial_image_dir[dir] = os.path.join(maindir, dir)
                print("with dir:{}".format(serial_image_dir[dir]))
    elif not os.path.exists(classes_test_data):
        print("serial list not exist........")
        return -1
    else:
        # 指定了车系就跑指定的车系文件的
        with open(classes_test_data, 'r') as f:
            while True:
                sid = f.readline()  # 整行读取数据
                if not sid:
                    break
                    pass

                sid = sid.replace('\n', '')
                if not os.path.exists(os.path.join(test_data_dir, sid)):
                    # 没有的就不放到返回集里面了
                    print("abort sid: " + sid + " case by file path not exist: " + os.path.join(test_data_dir, sid))
                    continue
                else:
                    serial_image_dir[sid] = os.path.join(test_data_dir, sid)
                    print("with dir:{}".format(serial_image_dir[sid]))

        f.close()

    # 赋值返回
    data['serial_image_dir'] = serial_image_dir
    data['result_output_path'] = args.result_output_path
    data['image_batch_size'] = args.image_batch_size
    # 创建输出文件夹
    if not os.path.exists(data['result_output_path']):
        os.mkdir(data['result_output_path'])
    # 创建错误日志
    # if not os.path.exists('%s' % err_logs_):
    #     os.mkdir(err_logs_)

    return data


def verify_h5model(args):
    data = prepareData(args)

    weights_path = args.weights_path
    test_data_dir = data['serial_image_dir']
    class_file_txt = args.class_file_txt
    classes_test_data = args.classes_test_data
    logs_path_ = args.result_output_path

    if not os.path.exists(logs_path_):
        os.mkdir(logs_path_)
    result_f = open(logs_path_ + 'keras_result.txt', 'a')

    try:
        classes_list = classes_test_data.split()
    except:
        classes_list = []

    with open(class_file_txt) as f:
        classes = f.readlines()
        class_count = len(classes)

    f.close()
    vehicleIds = [w.strip() for w in classes]

    # test data generator for prediction
    # test_datagen = ImageDataGenerator()
    # test_generator = test_datagen.flow_from_directory(
    #     test_data_dir,
    #     target_size=(img_width, img_height),
    #     batch_size=batch_size,
    #     shuffle=False,
    #     classes=classes_list,
    #     class_mode='categorical')
    #
    # test_image_list = test_generator.filenames

    # .h5文件包含model的load方法
    # inceptionV3_model = load_model(weights_path)

    # with tf.device("/cpu:0"):
    #     inception = InceptionV3(include_top=False, weights='imagenet',
    #                             input_tensor=None, pooling='avg', input_shape=(img_width, img_width, 3))
    #     output = inception.get_layer(index=-1).output  # shape=(None, 1, 1, 2048)
    #     output = Dropout(0.5)(output)
    #     output = Dense(class_count, activation='softmax', name='predictions')(output)
    #     model = Model(outputs=output, inputs=inception.input)
    #     for layer in inception.layers[:100]:
    #         layer.trainable = False
    with tf.device('/gpu:1'):
        densenet = DenseNet201(include_top=False, weights='imagenet',
                               input_tensor=None, input_shape=(img_width, img_width, 3), pooling='avg')
        output = densenet.get_layer(index=-1).output
        output = Dense(class_count, activation='softmax', name='predictions')(output)
        model = Model(outputs=output, inputs=densenet.input)
    #model = multi_gpu_model(model, gpus=2)

    model.load_weights(weights_path, by_name=True)

    # predictions = inceptionV3_model.predict_generator(test_generator, nbr_test_samples)
    for key, value in test_data_dir.items():
        result_f.flush()
        # 读取目录下的所有图片
        for maindir, subdir, file_name_list in os.walk(value):
            batch_size_ = data['image_batch_size']
            if maindir.find('/0') != -1:
                batch_size_ *= 3
            if batch_size_ > len(file_name_list):
                batch_size_ = len(file_name_list)

            # 获取配置的数码的图片
            for i in range(batch_size_):
                image_file = file_name_list[i]
                if image_file.find('.jpg') == -1 and image_file.find('.png') == -1 \
                        and image_file.find('.JPG') == -1:
                    print("not image file: {};{};".format(key, image_file))
                    continue

                try:
                    image_file_path = os.path.join(maindir, image_file)
                    print(image_file_path)
                except:
                    print("print error {}".format(sys.exc_info()))

                try:
                    # image_input = image.load_img(image_file_path, target_size=(299, 299))
                    image_input = scale_byRatio(image_file_path, return_width=img_width)
                    if image_input is None:
                        continue
                except:
                    print("load image error: {};{};".format(image_file_path, sys.exc_info()))
                    continue

                # height_tuple = (299, 299)
                # image_input = image_input.convert('RGB')
                # image_input = image_input.resize(height_tuple, Image.NEAREST)
                # image_input = image.img_to_array(image_input)
                # image_input /= 255.
                # image_input -= 0.5
                # image_input *= 2.
                # Add a 4th dimension for batch size (Keras)
                # i = image.img_to_array(image_input)
                p = np.array(image_input, dtype=np.float32)
                p = np.expand_dims(p, axis=0)
                p = preprocess_input(p)
                try:
                    prediction = model.predict(p)
                    node_id = np.argmax(prediction)
                    vehicleId = vehicleIds[node_id]
                    score = prediction[0][node_id]
                    print('{};{};{};{}'.format(key, vehicleId, score, image_file_path)
                          , file=result_f, flush=False)
                except IndexError:
                    print("IndexError: {};{};{}".format(image_file_path, node_id, prediction))
                except:
                    print("predict error: {};{};{}".format(image_file_path, node_id, prediction))

    result_f.flush()
    result_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights_path',
        type=str,
        default='/Users/ICD/Documents/source/mofang_UserActivity/UserActivityAnlyse/branches/tensorflow/server/milk_django/imgr/data/milk_InceptionV3_best_vehicleModel.h5',
        help='Path to the weights file.'
    )

    parser.add_argument(
        '--class_file_txt',
        type=str,
        default='/Users/ICD/Documents/source/mofang_UserActivity/UserActivityAnlyse/branches/tensorflow/server/milk_django/imgr/data/class_file.txt',
        help='The file contain the class in the model.'
    )

    parser.add_argument(
        '--test_data_dir',
        type=str,
        default='/Users/ICD/Documents/AI/testImg/milk/testSet/milk',
        help='Path to folders of images, include the subdirectory as the tag.'
    )

    parser.add_argument(
        '--classes_test_data',
        type=str,
        default='/Users/ICD/Documents/source/mofang_UserActivity/UserActivityAnlyse/branches/tensorflow/server/milk_django/imgr/data/class_file.txt',
        help='需要验证的子目录的列表,如果没有就跑test_data_dir下所有的子目录'
    )

    parser.add_argument(
        '--result_output_path',
        type=str,
        default='./logs/',
        help='the detail result and lumpsum result output path'
    )

    parser.add_argument(
        '--image_batch_size',
        type=int,
        default=30,
        help='How many image to test on in each serial.'
    )

    args, unknown = parser.parse_known_args(sys.argv[1:])

    verify_h5model(args)
