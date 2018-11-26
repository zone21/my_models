import argparse
import os
import time

import sys
import tensorflow as tf
from keras import Model
from keras.applications.densenet import DenseNet201, DenseNet169
from keras.backend import set_session
from keras.layers import Dense, K, Dropout
from keras.utils import multi_gpu_model
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.framework import graph_io

"""
Cover the keras InceptionV3 model with weights to the tensorflow pd model.
When cover the vehicle model to the web service, we found that keras load the h5 model file could not use multiple time
(unknown reason till now), so we need to use this method to switch the h5 model file to the pd model file, and then use
the tensorflow session to run. After that the service is okay.
"""

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.03
set_session(tf.Session(config=config))


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a prunned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    prunned so subgraphs that are not neccesary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def keras_to_tensorflow(args):
    input_fld = sys.path[0]
    weights_path = args.weights_path
    output_graph_name = time.strftime('%y%m%d', time.localtime()) + "_" + args.output_graph_name

    output_fld = input_fld + '/tensorflow_model/'

    class_count = len(open(args.class_file, 'r').readlines())
    print("class count {}".format(class_count))
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.03
    K.set_session(tf.Session(config=config))

    if args.model_json:
        from keras.models import model_from_json
        model = model_from_json(open(args.model_json).read())
    else:
        with tf.device("/cpu:0"):
            densenet = DenseNet201(include_top=False, weights='imagenet',
                                   input_tensor=None, input_shape=(299, 299, 3), pooling='avg')
            output = densenet.get_layer(index=-1).output
            output = Dropout(0.5)(output)
            output = Dense(class_count, activation='softmax', name='predictions')(output)
            model = Model(outputs=output, inputs=densenet.input)

    model = multi_gpu_model(model, gpus=2)
    model.load_weights(weights_path, by_name=True)

    print('input is :', model.input.name)
    print('output is:', model.output.name)

    sess = K.get_session()

    frozen_graph = freeze_session(K.get_session(), output_names=[model.output.op.name])

    graph_io.write_graph(frozen_graph, output_fld, output_graph_name, as_text=False)

    print('saved the constant graph (ready for inference) at: ', os.path.join(output_fld, output_graph_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights_path',
        type=str,
        default='/data0/shell/accuracy/model/vehicle/105_newcar_inceptionv3.h5',
        help='Path to the weights file.'
    )

    parser.add_argument(
        '--output_graph_name',
        type=str,
        default='vehicle_model_gpu_2.pb',
        help='The output model file name.'
    )

    parser.add_argument(
        '--class_file',
        type=str,
        default='/data0/shell/accuracy/model/vehicle/classes_file_105.txt',
        help='The class file contain all vehicle class'
    )

    parser.add_argument(
        '--model_json',
        type=str,
        default=None,
        help='the json file contain the model file'
    )

    args, unknown = parser.parse_known_args(sys.argv[1:])
    keras_to_tensorflow(args)
