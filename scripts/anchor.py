import numpy as np


def get_anchor_one_layer(img_shape,  sizes,
                         ratios, offset=0.5, dtype=np.float32):
    """Computer default anchor boxes for one feature layer.
    """
    # Compute the position grid: simple way.
    feat_shape = np.array(img_shape)
    feat_shape = feat_shape / 16
    step = 16
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    h = np.array(sizes)
    w = np.array(sizes)

    h = h * ratios[0][1]
    w = w * ratios[0][0]

    return y, x, h, w