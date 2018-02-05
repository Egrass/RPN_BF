"""loss function
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scripts import utils


def creat_mask(true_num, N, dtype):
    """ Create a [N,1]  mask, which has true_num 1 and N - true_num 0

    Args:
        true_num: the numbers of 'True' in bool mask
        N: the numbers of elements in bool mask

    """
    true_num = tf.cast(true_num, tf.int32)
    N = tf.cast(N, tf.int32)
    false_num = tf.cast(tf.subtract(N, true_num), tf.int32)

    mask1 = tf.ones([true_num])
    mask1 = tf.equal(mask1, 1)
    mask2 = tf.zeros([false_num])
    mask2 = tf.equal(mask2, 1)

    mask1 = tf.expand_dims(mask1, 1)
    mask2 = tf.expand_dims(mask2, 1)
    mask = tf.concat([mask1, mask2], axis=0)

    mask = tf.random_shuffle(mask)
    mask = tf.cast(mask, dtype)
    mask = tf.reduce_sum(mask, axis=1)
    return mask


def rpn_losses(logits, localisations, gclasses, glocalisations, gscores, max_match,
               max_threshold, min_threshold, num_classes, batch_size, negative_ratio, n_picture, lamb, scope='rpn_losses'):

    with tf.name_scope(scope, 'rpn_losses'):

        # Flatten out all vectors
        flogits = []
        flocalisations = []
        fgclasses = []
        fglocalisations = []
        fgscores = []
        fmax_match = []

        for i in range(1): # Anchors only base on one layer
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            fmax_match.append(tf.reshape(max_match[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        max_match = tf.concat(fmax_match, axis=0)
        dtype = logits.dtype

        # Compute positive matching mask
        pmask = gscores > max_threshold
        pmask = tf.logical_or(pmask, max_match)
        fpmask = tf.cast(pmask, dtype)
        all_positive = tf.reduce_sum(fpmask)

        # Compute negative matching mask
        nmask = tf.logical_and(tf.logical_not(pmask), gscores > -0.5)
        nmask = tf.logical_and(nmask, gscores < min_threshold)
        fnmask = tf.cast(nmask, dtype)
        all_negative = tf.reduce_sum(fnmask)

        # Compute positive and negative examples
        plogits = tf.boolean_mask(logits, pmask)
        pgclasses = tf.boolean_mask(gclasses, pmask)
        pgclasses = tf.cast(pgclasses, tf.int32)
        pgscores = tf.boolean_mask(gscores, pmask)
        plocalisations = tf.boolean_mask(localisations, pmask)
        pglocalisations = tf.boolean_mask(glocalisations, pmask)

        nlogits = tf.boolean_mask(logits, nmask)
        ngclasses = tf.boolean_mask(gclasses, nmask)
        ngclasses = tf.cast(ngclasses, tf.int32)
        ngscores = tf.boolean_mask(gscores, nmask)
        nlocalisations = tf.boolean_mask(localisations, nmask)
        nglocalisations = tf.boolean_mask(glocalisations, nmask)

        # Hard negative mining
        n_positive = n_picture / (1 + negative_ratio)
        n_positive = tf.minimum(tf.cast(n_positive, tf.int32), tf.cast(all_positive, tf.int32))
        n_negative = tf.cast(n_picture - n_positive, tf.int32)

        # Compute the whole numbers
        n_positive = n_positive * batch_size
        n_negative = n_negative * batch_size
        n_picture = n_picture * batch_size

        # Random mask
        pmask = creat_mask(n_positive, all_positive, dtype)
        nmask = creat_mask(n_negative, all_negative, dtype)

        # Add cross-entropy loss
        with tf.name_scope('cross_entropy_pos'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=plogits,
                                                                  labels=pgclasses)
            cross_entropy_pos = tf.div(tf.reduce_sum(loss * pmask), n_picture, name='value')
            tf.add_to_collection('losses', cross_entropy_pos)

        with tf.name_scope('cross_entropy_neg'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=nlogits,
                                                                  labels=ngclasses)
            cross_entropy_neg = tf.div(tf.reduce_sum(loss * nmask), n_picture, name='value')
            tf.add_to_collection('losses', cross_entropy_neg)

        with tf.name_scope('localization'):
            # Weights Tensor
            loss = utils.abs_smooth(plocalisations - pglocalisations)
            loss = tf.reduce_sum(loss, axis=1)
            localization = tf.div(tf.reduce_sum(loss * pmask), tf.cast(n_positive, tf.float32), name='value')
            lamb = tf.cast(lamb, tf.float32)
            localization = localization * lamb
            tf.add_to_collection('losses', localization)

        with tf.name_scope('regularization_loss'):
            regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
            tf.add_to_collection('losses', regularization_loss)


        return tf.add_n(tf.get_collection('losses'), name='total_loss')