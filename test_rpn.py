from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
import math
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow.contrib.slim as slim


from scripts import config
from scripts import anchor
from scripts import preprocess
from scripts import encode
from scripts import utils
from scripts import net
from scripts import losses
from dataset import read_dataset


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './eval_log',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', './data',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './log_new',
                           """Directory where to read model checkpoints.""")


def evaluate(C):
    with tf.Graph().as_default() as g:
        # Get image and labels
        # Get the format of tfrecord
        dataset = read_dataset.get_split(C.eval_data, C.train_images, C.test_images)

        # Create a dataset provider and batches
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=4,
            common_queue_capacity=20 * C.batch_size,
            common_queue_min=10 * C.batch_size,
            shuffle=True
        )

        [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                         'object/label',
                                                         'object/bbox'])

        # Preprocess image, labels and bboxes
        image, glabels, gbboxes = preprocess.preprocess_for_test(image, glabels, gbboxes, C.resize_size, data_format='NHWC')
        image = tf.expand_dims(image, 0)

        # Get the result
        localisations, logits = net.rpn_net(image, C)
        localisations = tf.reshape(localisations, [-1, 4])
        logits = tf.reshape(logits, [-1, C.num_classes])

        # Choose some prediction
        mask = tf.greater(logits[:, 1], C.nms_overlap)
        localisations = tf.boolean_mask(localisations, mask)
        logits = tf.boolean_mask(logits, mask)

        # Saver for importing variables
        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(C.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            shape = sess.run(shape)
            print(shape)
            # Start the queue runners.
            # coord = tf.train.Coordinator()
            # try:
            #     threads = []
            #     for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            #         threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
            #                                          start=True))
            #
            #     num_iter = int(math.ceil(C.test_images / C.batch_size))
            #     true_count = 0  # Counts the number of correct predictions.
            #     total_sample_count = num_iter * FLAGS.batch_size
            #     step = 0
            #     while step < num_iter and not coord.should_stop():
            #         predictions = sess.run([top_k_op])
            #         true_count += np.sum(predictions)
            #         step += 1
            #         # Write down the results
            #         #print_result(localisations, logits)
            #
            #     # Compute precision @ 1.
            #     precision = true_count / total_sample_count
            #     print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
            #
            #
            # except Exception as e:  # pylint: disable=broad-except
            #     coord.request_stop(e)
            #
            # coord.request_stop()
            # coord.join(threads, stop_grace_period_secs=10)


def main(argv=None):  # pylint: disable=unused-argument
    C = config.Config()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    C.eval_dir = FLAGS.eval_dir
    C.eval_data = FLAGS.eval_data
    C.checkpoint_dir = FLAGS.checkpoint_dir
    evaluate(C)


if __name__ == '__main__':
    tf.app.run()
