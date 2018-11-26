# -*- coding: utf-8 -*-

from math import ceil
import numpy as np
from keras.applications.densenet import DenseNet201
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Dropout, merge
from keras.models import Model
from keras.optimizers import *
from utils_dist import SGDRScheduler, CustomModelCheckpoint, SequenceData
from keras.callbacks import TensorBoard
from keras.preprocessing import image
from keras.utils import training_utils

import os
# import horovod.keras as hvd
np.random.seed(1024)

with open('./data/class_134.txt') as f:
    classes = f.readlines()

filelist = [w.strip() for w in classes]

FINE_TUNE = False
warmup_epochs = 1
base_lr = 0.0125
momentum = 0.9
LEARNING_RATE = 0.003
NBR_EPOCHS = 1500
BATCH_SIZE = 22
IMG_WIDTH = 299
IMG_HEIGHT = 299
monitor_index = 'val_acc'
NBR_MODELS = len(filelist)
USE_CLASS_WEIGHTS = False
RANDOM_SCALE = True
multi_gpu = True
encoding = "GB18030"
resume_from_epoch = 0

best_model_file = "./data/DenseNet201_best_vehicleModel.h5"
weight_path = './data/InceptionV3_best_vehicleModel_404_20181006.h5'
train_path = './data/train_file_134.txt'
val_path = './data/val_file_134.txt'

if __name__ == "__main__":
    # Horovod: initialize Horovod.
    # hvd.init()

    # Pin a server GPU to be used by this process
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # local_rank = hvd.local_rank()
    # if not multi_gpu:
    #     config.gpu_options.visible_device_list = str(local_rank)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.9
    K.set_session(tf.Session(config=config))

    # ['/job:localhost/replica:0/task:0/device:CPU:0', '/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1']
    nbr_gpus = len(training_utils._get_available_devices()) - 1

    # rank = hvd.rank()
    # verbose = 1 if rank == 0 else 0
    # print('verbose is %d' % verbose)
    # print('hvd.rank() is %d' % rank)
    # resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')

    print('Loading InceptionV3 Weights ...')
    with tf.device('/cpu:0'):
        densenet = DenseNet201(include_top=False, weights='imagenet',
                               input_tensor=None, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), pooling='avg')
        output = densenet.get_layer(index=-1).output
        output = Dropout(0.5)(output)
        output = Dense(NBR_MODELS, activation='softmax', name='predictions')(output)
        model = Model(outputs=output, inputs=densenet.input)

    model = training_utils.multi_gpu_model(model, gpus=nbr_gpus)

    if FINE_TUNE:
        model.load_weights(best_model_file)
    print('Training model begins...')
    # BATCH_SIZE *= nbr_gpus

    # optimizer = SGD(lr=base_lr * hvd.size(), momentum=momentum)
    # hvd_size = hvd.size()
    optimizer = SGD(lr=LEARNING_RATE * nbr_gpus, momentum=0.9, decay=0.0, nesterov=True)
    #optimizer = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    # Horovod: add Horovod Distributed Optimizer.
    # hvd_optimizer = hvd.DistributedOptimizer(optimizer)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    #model.compile(loss=amsoftmax_loss, optimizer=optimizer, metrics=["accuracy"])

    # autosave best Model
    best_model = CustomModelCheckpoint(model, best_model_file, monitor_index=monitor_index)
    reduce_lr = ReduceLROnPlateau(monitor=monitor_index, factor=0.5, patience=10 // nbr_gpus, verbose=1, min_lr=0.0001*nbr_gpus)
    early_stop = EarlyStopping(monitor=monitor_index, patience=(50 // nbr_gpus), verbose=1, min_delta=0.001)

    # 准备数据 start
    try:
        train_data_lines = open(train_path, 'r', encoding=encoding).readlines()
    except UnicodeDecodeError:
        train_data_lines = open(train_path, 'r', encoding='UTF-8').readlines()
    train_data_lines = [w.strip() for w in train_data_lines if os.path.exists(w.strip().split(' ')[0])]
    # The train_data_lines must be sampled 1/n for each node because use distributed in multiple nodes
    # lines_ = ceil(len(train_data_lines) / hvd_size)
    # node_train_data_lines = train_data_lines[(lines_ * rank):(lines_ * (rank+1))]
    node_train_data_lines = train_data_lines
    node_nbr_train = len(node_train_data_lines)
    print('# Train Images: {}.'.format(node_nbr_train))
    steps_per_epoch = int(ceil(node_nbr_train / (BATCH_SIZE*nbr_gpus)))

    try:
        val_data_lines = open(val_path, 'r', encoding=encoding).readlines()
    except UnicodeDecodeError:
        val_data_lines = open(val_path, 'r', encoding='UTF-8').readlines()
    val_data_lines = [w.strip() for w in val_data_lines if os.path.exists(w.strip().split(' ')[0])]
    # The val_data_lines must be sampled 1/n for each node because use distributed in multiple nodes
    # val_lines_ = ceil(len(val_data_lines) / hvd_size)
    # node_val_data_lines = val_data_lines[(val_lines_ * rank):(val_lines_ * (rank+1))]
    node_val_data_lines = val_data_lines
    node_nbr_val = len(node_val_data_lines)
    print('# Val Images: {}.'.format(node_nbr_val))
    validation_steps = int(ceil(node_nbr_val / (BATCH_SIZE*nbr_gpus)))
    # 准备数据 end

    callbacks = [best_model, reduce_lr, early_stop, TensorBoard(log_dir='./tb_log')]

    # Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.

    model.fit_generator(SequenceData(node_train_data_lines, nbr_classes=NBR_MODELS,
                                     batch_size=BATCH_SIZE, img_width=IMG_WIDTH,
                                     img_height=IMG_HEIGHT, random_scale=RANDOM_SCALE,
                                     shuffle=True, augment=True),
                        steps_per_epoch=steps_per_epoch, epochs=NBR_EPOCHS, verbose=1,
                        validation_data=SequenceData(node_val_data_lines, nbr_classes=NBR_MODELS,
                                                     batch_size=BATCH_SIZE, img_width=IMG_WIDTH,
                                                     img_height=IMG_HEIGHT, random_scale=RANDOM_SCALE,
                                                     shuffle=True, augment=False),
                        validation_steps=validation_steps,
                        callbacks=callbacks,
                        max_queue_size=128, workers=2,use_multiprocessing=True)
