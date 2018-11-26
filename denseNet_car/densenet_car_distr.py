#coding=GB18030

from math import ceil
import numpy as np
import time
from keras.applications.densenet import DenseNet201, DenseNet169
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Dropout, merge
from keras.models import Model
from keras.optimizers import *
from utils_dist import SGDRScheduler, CustomModelCheckpoint, SequenceData
from keras.callbacks import TensorBoard
from keras.preprocessing import image
from keras.utils import training_utils

import os
import horovod.keras as hvd
np.random.seed(1024)

with open('./data/624/class_624.txt') as f:
    NBR_MODELS = len(f.readlines())

FINE_TUNE = True
warmup_epochs = 1
base_lr = 0.0125
momentum = 0.9
LEARNING_RATE = 0.001
NBR_EPOCHS = 1500
BATCH_SIZE = 16
IMG_WIDTH = 299
IMG_HEIGHT = 299
monitor_index = 'acc'
USE_CLASS_WEIGHTS = False
RANDOM_SCALE = True
encoding = "gbk"
resume_from_epoch = 0

train_path = './data/624/train_file_624.txt'
val_path = './data/624/val_file_624.txt'
new_classes = ''
best_model_file = "./data/624/"
last_mt_time = 0
##get the last modify h5 file
for maindir, subdirs, file_name_list in os.walk(best_model_file):
    for file in file_name_list:
        if file.find('.h5') == -1:
            continue

        mt_time = os.path.getmtime(os.path.join(maindir, file))
        if last_mt_time < mt_time:
            last_mt_time = mt_time
            best_model_file = os.path.join(maindir, file)

if __name__ == "__main__":
    # Horovod: initialize Horovod.
    hvd.init()

    # Pin a server GPU to be used by this process
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    local_rank = hvd.local_rank()
    config.gpu_options.visible_device_list = str(local_rank)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    K.set_session(tf.Session(config=config))

    # ['/job:localhost/replica:0/task:0/device:CPU:0', '/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1']
    nbr_gpus = len(training_utils._get_available_devices()) - 1

    rank = hvd.rank()
    verbose = 1 if rank == 0 else 0
    print('verbose is %d' % verbose)
    print('hvd.rank() is %d' % rank)
    resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')

    densenet = DenseNet201(include_top=False, weights='imagenet',
                           input_tensor=None, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), pooling='avg')
    densenet.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy',
                          metric='accuracy')

    output = densenet.get_layer(index=-1).output
    output = Dropout(0.3)(output)
    output = Dense(NBR_MODELS, activation='softmax', name='predictions')(output)
    model = Model(outputs=output, inputs=densenet.input)

    if FINE_TUNE:
        print('Loading DenseNet201 Weights in file %s' % best_model_file)
        model.load_weights(best_model_file)

        if new_classes:
            with open(new_classes) as f:
                NBR_MODELS = len(f.readlines())

            f.close()
            print('use fine tune.....')
            output = model.get_layer(index=-2).output
            output = Dense(NBR_MODELS, activation='softmax', name='predictions')(output)
            model = Model(outputs=output, inputs=densenet.input)

    print('Training model begins...')
    # BATCH_SIZE *= nbr_gpus

    # optimizer = SGD(lr=base_lr * hvd.size(), momentum=momentum)
    hvd_size = hvd.size()
    optimizer = SGD(lr=LEARNING_RATE * hvd_size, momentum=0.9, decay=0.0, nesterov=True)
    #optimizer = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    # Horovod: add Horovod Distributed Optimizer.
    hvd_optimizer = hvd.DistributedOptimizer(optimizer)

    model.compile(loss="categorical_crossentropy", optimizer=hvd_optimizer, metrics=["accuracy"])
    #model.compile(loss=amsoftmax_loss, optimizer=optimizer, metrics=["accuracy"])

    # autosave best Model
    #out_model_name = './data/' + str(NBR_MODELS) + '/InceptionV3_best_vehicleModel_' + time.strftime('%y%m%d', time.localtime()) + ".h5"
    out_model_name = './data/624' + '/DenseNet201_best_vehicleModel_' + time.strftime('%y%m%d', time.localtime()) + ".h5"
    # out_model_name = './data/624/DenseNet201_best_vehicleModel_181107.h5'
    best_model = CustomModelCheckpoint(model, out_model_name, monitor_index=monitor_index)
    reduce_lr = ReduceLROnPlateau(monitor=monitor_index, factor=0.5, patience=10 // hvd_size, verbose=1, min_lr=0.001*hvd_size)
    early_stop = EarlyStopping(monitor=monitor_index, patience=(50 // hvd_size), verbose=1, min_delta=0.001)

    # 准备数据 start
    try:
        train_data_lines = open(train_path, 'r', encoding=encoding).readlines()
    except UnicodeDecodeError:
        train_data_lines = open(train_path, 'r', encoding=encoding).readlines()
    train_data_lines = [w.strip() for w in train_data_lines if os.path.exists(w.strip().split(' ')[0])]
    # The train_data_lines must be sampled 1/n for each node because use distributed in multiple nodes
    node_train_data_lines = train_data_lines
    node_nbr_train = len(node_train_data_lines)
    print('# Train Images: {}.'.format(node_nbr_train))
    steps_per_epoch = int(ceil(node_nbr_train / (BATCH_SIZE*hvd_size)))

    try:
        val_data_lines = open(val_path, 'r', encoding=encoding).readlines()
    except UnicodeDecodeError:
        val_data_lines = open(val_path, 'r', encoding=encoding).readlines()
    val_data_lines = [w.strip() for w in val_data_lines if os.path.exists(w.strip().split(' ')[0])]
    # The val_data_lines must be sampled 1/n for each node because use distributed in multiple nodes
    node_val_data_lines = val_data_lines
    node_nbr_val = len(node_val_data_lines)
    print('# Val Images: {}.'.format(node_nbr_val))
    scale_up = 4
    validation_steps = int(ceil(node_nbr_val / (BATCH_SIZE*hvd_size*scale_up)))
    # 准备数据 end

    callbacks = list()
    # Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
    if rank == 0:
        callbacks.append(best_model)
        callbacks.append(early_stop)
        callbacks.append(TensorBoard(log_dir='./tb_log', histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True,
                                     write_grads=False, write_images=False))

    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    # Horovod: average metrics among workers at the end of every epoch.
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard, or other metrics-based callbacks.
    callbacks.append(hvd.callbacks.MetricAverageCallback())
    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=warmup_epochs, verbose=verbose))
    callbacks.append(reduce_lr)

    validation_data = SequenceData(node_val_data_lines, nbr_classes=NBR_MODELS,
                 batch_size=BATCH_SIZE * scale_up, img_width=IMG_WIDTH,
                 img_height=IMG_HEIGHT, random_scale=RANDOM_SCALE,
                 shuffle=True, augment=False)

    model.fit_generator(SequenceData(node_train_data_lines, nbr_classes=NBR_MODELS,
                                     batch_size=BATCH_SIZE, img_width=IMG_WIDTH,
                                     img_height=IMG_HEIGHT, random_scale=RANDOM_SCALE,
                                     shuffle=True, augment=True),
                        steps_per_epoch=steps_per_epoch, epochs=NBR_EPOCHS, verbose=verbose,
                        validation_data=validation_data,
                        validation_steps=validation_steps,
                        callbacks=callbacks,
                        max_queue_size=128, workers=2, use_multiprocessing=True)

