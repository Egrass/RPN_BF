import tensorflow as tf
import tensorflow.contrib.slim as slim
import os


def get_split(dataset_dir, train_images, test_images,
              file_pattern=None, reader=None, num_classes=2):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.
    """

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    split_to_sizes = {
        'train': train_images,
        'test': test_images,
    }

    items_to_descriptions = {
        'image': 'A color image of varying height and width.',
        'shape': 'Shape of the image',
        'object/bbox': 'A list of bounding boxes, one per each object.',
        'object/label': 'A list of labels, one per each object.',
    }

    if file_pattern == None:
        file_pattern = []
        file_list = os.listdir(dataset_dir)
        for file in file_list:
            path = os.path.join(dataset_dir, file)
            file_pattern.append(path)
    else:
        file_pattern = os.path.join(dataset_dir, file_pattern)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=split_to_sizes,
        items_to_descriptions=items_to_descriptions,
        num_classes=num_classes,
        labels_to_names=None)
