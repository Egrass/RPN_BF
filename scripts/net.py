""""RPN net definition"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scripts import rpn_arg_scope


def rpn_net(inputs, C, if_training=True, scope='rpn_net'):
    num_anchors = len(C.anchor_box_ratios) * len(C.anchor_box_scales)
    end_points = {}

    arg_scope = rpn_arg_scope.rpn_arg_scope(weight_decay=C.weight_decay)
    with slim.arg_scope(arg_scope):
        with tf.variable_scope(scope, 'rpn_net', [inputs], reuse=None):
            # VGG-16 blocks
            # block1
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            # block2
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            # block3
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            # end_points['block3'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            # block4
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            # end_points['block4'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool4')

            # block5
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            end_points['block5'] = net

            with tf.variable_scope('rpn'):
                net = slim.conv2d(net, 512, [3, 3], scope='conv_proposal')
                bbox_pred = slim.conv2d(net, num_anchors*4, [1, 1], scope='bbox_pred')
                cls_score = slim.conv2d(net, num_anchors*2, [1, 1], scope='cls_score')

        return bbox_pred, cls_score