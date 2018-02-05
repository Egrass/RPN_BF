from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
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

tf.app.flags.DEFINE_string('train_dir', './log',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('dataset_dir', './tfrecord',
                           """Directory where to find data""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def tower_loss(C, scope, images, gclasses, glocalisations, gscores, max_match):
    """Calculate the total loss on a single tower running the CIFAR model.

    Args:
        scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
        images: Images. 4D tensor of shape [batch_size, height, width, 3].
        labels: Labels. 1D tensor of shape [batch_size].

    Returns:
        Tensor of shape [] containing the total loss for a batch of data
    """

    # Build inference Graph.
    localisations, logits = net.rpn_net(images, C)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = losses.rpn_losses(logits, localisations, gclasses, glocalisations, gscores, max_match,
               C.gt_p, C.gt_ng, C.num_classes, C.batch_size, C.negative_ratio, C.n_picture, C.lamb)

    # Assemble all of the losses for the current tower only.
    all_lose = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(all_lose, name='total_loss')

    for l in all_lose + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = l.op.name
        tf.summary.scalar(loss_name, l)

    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train(C):
    """"Train with RPN net"""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule
        decay_steps = C.learning_rate[1][0]
        initial_lr = C.learning_rate[0][1]
        decay_factor = C.learning_rate[1][1] / C.learning_rate[0][1]
        lr = tf.train.exponential_decay(initial_lr,
                                        global_step,
                                        decay_steps,
                                        decay_factor,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.GradientDescentOptimizer(lr)

        # Get the format of tfrecord
        dataset = read_dataset.get_split(C.dataset_dir, C.train_images, C.test_images)

        # Get the network and its anchors
        rpn_anchors = anchor.get_anchor_one_layer(C.resize_size, C.anchor_box_scales, C.anchor_box_ratios)

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
        image, glabels, gbboxes = preprocess.preprocess_for_train(image, glabels, gbboxes, C.resize_size, data_format='NHWC')

        # Encode groundtruth labels and bboxes
        gclasses, glocalisations, gscores, max_match = encode.rpn_encode_one_layer(glabels, gbboxes, rpn_anchors, C.prior_scaling)
        batch_shape = [1] + [1] * 4

        # Training batches and queue
        r = tf.train.batch(
            utils.reshape_list([image, gclasses, glocalisations, gscores, max_match]),
            batch_size=C.batch_size,
            num_threads=4,
            capacity=5 * C.batch_size
        )
        b_image, b_gclasses, b_glocalisations, b_gscores, b_max_match = utils.reshape_list(r, batch_shape)

        # Intermediate queueing: unique batch computation pipeline for all
        # GPUs running the training.
        batch_queue = slim.prefetch_queue.prefetch_queue(utils.reshape_list([b_image, b_gclasses,
                                                                             b_glocalisations, b_gscores, b_max_match]),
                                                         capacity=2 * 1)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(C.gpu_nums):
                with tf.device('/gpu:%d' % i):
                #with tf.device('/cpu'):
                    with tf.name_scope('tower_%d' % i) as scope:
                        # Dequeue one batch for the GPU
                        images, gclasses, glocalisations, gscores, max_match = utils.reshape_list(
                            batch_queue.dequeue(), batch_shape
                        )

                        # Construct network
                        loss = tower_loss(C, scope, images, gclasses, glocalisations, gscores, max_match)

                        # Reuse  variables for the next tower
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
          if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Group all updates to into a single train op
        train_op = tf.group(apply_gradient_op)

        # Create a saver
        saver = tf.train.Saver(tf.global_variables())

        # Build the sumary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))
        sess.run(init)

        # Start the queue runners
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(C.train_dir, sess.graph)

        for step in xrange(C.max_steps):
            _, loss_value = sess.run([train_op, loss])

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                format_str = ('%s: step %d, loss = %.2f')
                print(format_str % (datetime.now(), step, loss_value))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == C.max_steps:
                checkpoint_path = os.path.join(C.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    C = config.Config()
    C.dataset_dir = FLAGS.dataset_dir
    C.train_dir = FLAGS.train_dir
    train(C)


if __name__ == '__main__':
    tf.app.run()