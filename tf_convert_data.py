import tensorflow as tf
import os
from dataset import caltech_to_tfrecord

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'dataset_dir', './data',
    'Directory where the original dataset is stored.')
tf.app.flags.DEFINE_string(
    'output_dir', './tfrecord',
    'Output directory where to store TFRecords files.')

def main(argv=None):
    print('Dataset directory:', FLAGS.dataset_dir)
    print('Output directory:', FLAGS.output_dir)
    images_path = os.path.join(FLAGS.dataset_dir, 'images')
    annotations_path = os.path.join(FLAGS.dataset_dir, 'annotations')
    if not tf.gfile.Exists(images_path):
        raise ValueError('Not found images directory.')
    if not tf.gfile.Exists(annotations_path):
        raise ValueError('Not found annotations directory')
    caltech_to_tfrecord.run(images_path, annotations_path, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()