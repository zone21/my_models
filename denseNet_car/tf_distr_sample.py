import argparse
import sys

import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dropout, Dense

FLAGS = None

"""Constructor.

Sets the properties `cluster_spec`, `is_chief`, `master` (if `None` in the
args), `num_ps_replicas`, `task_id`, and `task_type` based on the
`TF_CONFIG` environment variable, if the pertinent information is
present. The `TF_CONFIG` environment variable is a JSON object with
attributes: `cluster`, `environment`, and `task`.

`cluster` is a JSON serialized version of `ClusterSpec`'s Python dict from
`server_lib.py`, mapping task types (usually one of the TaskType enums) to a
list of task addresses.

`environment` specifies the runtime environment for the job (usually one of
the `Environment` enums). Defaults to `LOCAL`.

`task` has two attributes: `type` and `index`, where `type` can be any of
the task types in `cluster`. When `TF_CONFIG` contains said information, the
following properties are set on this class:

* `task_type` is set to `TF_CONFIG['task']['type']`. Defaults to `None`.
* `task_id` is set to `TF_CONFIG['task']['index']`. Defaults to 0.
* `cluster_spec` is parsed from `TF_CONFIG['cluster']`. Defaults to {}.
* `master` is determined by looking up `task_type` and `task_id` in the
  `cluster_spec`. Defaults to ''.
* `num_ps_replicas` is set by counting the number of nodes listed
  in the `ps` attribute of `cluster_spec`. Defaults to 0.
* `num_worker_replicas` is set by counting the number of nodes listed
  in the `worker` attribute of `cluster_spec`. Defaults to 0.
* `is_chief` is deteremined based on `task_type`, `type_id`, and
  `environment`.

Example:
```
  cluster = {'ps': ['host1:2222', 'host2:2222'],
             'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
  os.environ['TF_CONFIG'] = json.dumps(
      {'cluster': cluster,
       'task': {'type': 'worker', 'index': 1}})
  config = ClusterConfig()
  assert config.master == 'host4:2222'
  assert config.task_id == 1
  assert config.num_ps_replicas == 2
  assert config.num_worker_replicas == 3
  assert config.cluster_spec == server_lib.ClusterSpec(cluster)
  assert config.task_type == 'worker'
  assert not config.is_chief
```

Args:
  master: TensorFlow master. Defaults to empty string for local.
  evaluation_master: The master on which to perform evaluation.
"""


def main(_):
    """
    python trainer.py \
        --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
        --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
        --job_name=ps --task_index=0
    # On ps1.example.com:
    python trainer.py \
        --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
        --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
        --job_name=ps --task_index=1
    # On worker0.example.com:
    python trainer.py \
        --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
        --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
        --job_name=worker --task_index=0
    # On worker1.example.com:
    python trainer.py \
        --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
        --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
        --job_name=worker --task_index=1
    """

    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")
    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    # Get the number of workers.
    num_workers = len(worker_hosts)

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    # 如果使用gpu
    num_gpus = FLAGS.num_gpus
    if num_gpus > 0:
        # Avoid gpu allocation conflict: now allocate task_num -> #gpu
        # for each worker in the corresponding machine
        gpu = (FLAGS.task_index % num_gpus)
        # 分配worker到指定gpu上运行
        worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
    # 如果使用cpu
    elif num_gpus == 0:
        # Just allocate the CPU to worker server
        # 把cpu分配给worker
        cpu = 0
        worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device=worker_device,
                cluster=cluster)):

            # Build model...
            # loss = ...
            # global_step = tf.contrib.framework.get_or_create_global_step()
            #
            # train_op = tf.train.AdagradOptimizer(0.01).minimize(
            #     loss, global_step=global_step)
            # Instantiate a Keras inception v3 model.
            keras_densenet = tf.keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet',
                               input_tensor=None, input_shape=(FLAGS.img_width, FLAGS.img_height, 3), pooling='avg')
            # Compile model with the optimizer, loss, and metrics you'd like to train with.
            sgd = tf.keras.optimizers.SGD(lr=FLAGS.learning_rate * num_gpus, momentum=0.9, decay=0.0, nesterov=True)
            output = keras_densenet.get_layer(index=-1).output
            output = Dropout(0.5)(output)
            output = Dense(NBR_MODELS, activation='softmax', name='predictions')(output)
            keras_model = Model(outputs=output, inputs=keras_densenet.input)

            keras_model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics='accuracy')
            # Create an Estimator from the compiled Keras model. Note the initial model
            # state of the keras model is preserved in the created Estimator.
            est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model=keras_model)

            # Treat the derived Estimator as you would with any other Estimator.
            # First, recover the input name(s) of Keras model, so we can use them as the
            # feature column name(s) of the Estimator input function:
            input_name = keras_densenet.input_names  # print out: ['input_1']
            # Once we have the input name(s), we can create the input function, for example,
            # for input(s) in the format of numpy ndarray:
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={input_name: train_data},
                y=train_labels,
                num_epochs=1,
                shuffle=False)
            # To train, we call Estimator's train function:
            tf.estimator.train_and_evaluate(
                est_inception_v3,
                train_input_fn,
                train_input_fn
            )

        # The StopAtStepHook handles stopping after running given steps.
        hooks = [tf.train.StopAtStepHook(last_step=1000000)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir="/tmp/train_logs",
                                               hooks=hooks) as mon_sess:
            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See <a href="./../api_docs/python/tf/train/SyncReplicasOptimizer"><code>tf.train.SyncReplicasOptimizer</code></a> for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                mon_sess.run(train_op)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    # Flags for defining the number of gpu
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="Total number of gpus for each machine.If you don't use GPU, please set it to '0'"
    )
    # Flags for defining the learning rate LEARNING RATE
    parser.add_argument(
        "--learning_rate",
        type=int,
        default=0.001,
        help="learning rate for the model define"
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=299,
        help="img width for the model input"
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=299,
        help="img height for the model input"
    )
    parser.add_argument(
        "--class_file",
        type=int,
        default=299,
        help="the file contain all the classification for the model output"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
