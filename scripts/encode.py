"""Encode groundtruth labels and bounding boxes using rpn net anchors
"""
import tensorflow as tf
import numpy as np


def rpn_encode_one_layer(labels, bboxes, anchors, prior_scaling, dtype=tf.float32):
    """Encode groundtruth labels and bounding boxes using rpn anchors from
    one layer.

    Arguments:
        labels: 1D Tensor(int64) containing groundtruth labels;
        bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
        anchors_layer: Numpy array with layer anchors;
        matching_threshold: Threshold for positive match with groundtruth bboxes;
        prior_scaling: Scaling of encoded coordinates.

    Return:
        (target_labels, target_localizations, target_scores): Target Tensors.
    """
    # Anchors coordinates and volume
    yref, xref, href, wref = anchors
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + href / 2.
    vol_anchors = (xmax - xmin) * (ymax - ymin)

    # Initialize
    shape = (yref.shape[0], yref.shape[1], href.size)
    feat_labels = tf.zeros(shape, dtype=tf.int64)
    feat_scores = tf.zeros(shape, dtype=dtype)

    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    max_match = tf.zeros(shape, dtype=dtype)
    max_match = tf.greater(max_match, 1) # creat logical tensor

    def jaccard_with_anchors(bbox):
        """Compute jaccard score between a box and the anchors.
        """

        a = bbox[0]
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol \
            + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        jaccard = tf.div(inter_vol, union_vol)
        return jaccard

    def condition(i, feat_labels, feat_scores,
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax, max_match):
        """Condition: check label index.
        """
        r = tf.less(i, tf.shape(labels))
        return r[0]

    def body(i, feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax, max_match):
        """Body: update feature labels, scores and bboxes.
        """
        # Jaccard score
        label = labels[i]
        bbox = bboxes[i]
        jaccard = jaccard_with_anchors(bbox)

        MAX = tf.reduce_max(jaccard)
        temp = tf.equal(jaccard, MAX)
        max_match = tf.logical_or(temp, max_match) # set the best Iou Box to positive

        mask = tf.greater(jaccard, feat_scores)
        mask = tf.logical_and(mask, feat_scores > -0.5)
        mask = tf.logical_and(mask, label < 2)
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, dtype)

        # Update values using mask
        feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = tf.where(mask, jaccard, feat_scores)
        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

        return [i+1, feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax, max_match]


    # Main loop definition
    i = 0
    [i, feat_labels, feat_scores,
    feat_ymin, feat_xmin, feat_ymax, feat_xmax, max_match] = tf.while_loop(condition, body,
                                                                         [i, feat_labels, feat_scores,
                                                                          feat_ymin, feat_xmin, feat_ymax, feat_xmax, max_match])

    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    # Encode features.
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_w = feat_xmax - feat_xmin
    feat_h = tf.log(feat_h / href) / prior_scaling[2]
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    return feat_labels, feat_localizations, feat_scores, max_match