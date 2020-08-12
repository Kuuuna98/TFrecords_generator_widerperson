from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class DatasetGenerator(object):
    def __init__(self, data_dir, input_size, output_size, batch_size):
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self._num_bbox = 0

    def run(self, train_num_bbox, val_num_bbox, preprocessed=None):
        return self.get_tfdataset(
            train=True, num_bbox=train_num_bbox,
            preprocessed=preprocessed), self.get_tfdataset(
                train=False, num_bbox=val_num_bbox)

    def get_tfdataset(self, train, num_bbox, preprocessed=None):
        self._num_bbox = num_bbox
        tfrecords_dir = self.data_dir / 'tfrecords' / f'{"train" if train else "val"}_{self.input_size}'
        tfrecords_files = [str(f) for f in tfrecords_dir.glob('*.tfrecords')]
        tfrecords = tf.data.TFRecordDataset(
            tfrecords_files, num_parallel_reads=len(tfrecords_files))
        ds = tfrecords
        if train:
            ds = ds.shuffle(buffer_size=10000, seed=42)
        ds = ds.map(self._parse_tfrecord,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if train and preprocessed:

            ds = ds.map(self._extract_valid_bboxes,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

            ds = ds.map(self._convert_image_float32,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

            for preprocess in preprocessed:
                ds = ds.map(preprocess,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

            ds = ds.map(self._convert_image_uint8,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

            ds = ds.map(self._pad_bboxes,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds = ds.map(lambda i, o: (preprocess_input(tf.cast(i, tf.float32)), o),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds

    def _parse_tfrecord(self, tfrecord):
        tf_example = tf.io.parse_single_example(tfrecord,
                                                features={
                                                    'image_raw':
                                                    tf.io.FixedLenFeature(
                                                        (), tf.string),
                                                    'bboxes':
                                                    tf.io.FixedLenFeature(
                                                        (self._num_bbox, 4),
                                                        tf.float32)
                                                })
        image = tf.image.decode_jpeg(tf_example['image_raw'], channels=3)
        bboxes = tf_example['bboxes']

        return image, bboxes

    def _convert_image_float32(self, image, bboxes):
        return tf.image.convert_image_dtype(image, tf.float32), bboxes

    def _convert_image_uint8(self, image, bboxes):
        return tf.image.convert_image_dtype(image, tf.uint8), bboxes

    def _extract_valid_bboxes(self, image, bboxes):
        valid_bboxes = bboxes[tf.reduce_any(bboxes != 0, axis=-1)]
        return image, valid_bboxes

    def _pad_bboxes(self, image, bboxes):
        padded_bboxes = tf.pad(
            bboxes, [[0, self._num_bbox - tf.shape(bboxes)[0]], [0, 0]],
            'CONSTANT')
        return image, padded_bboxes


if __name__ == '__main__':
    from absl import app
    from absl import flags

    FLAGS = flags.FLAGS

    flags.DEFINE_string('data_dir', 'data', 'Data root directory path')
    flags.DEFINE_string('data_format', 'channels_last',
                        'Model input data format')
    flags.DEFINE_integer('input_size', 512, 'Model input size')
    flags.DEFINE_integer('batch_size', 16, 'Batch size')

    flags.DEFINE_integer('train_num_bbox', 660, 'train_num_bbox')
    flags.DEFINE_integer('val_num_bbox', 256, 'val_num_bbox')

    def main(_):
        tf.keras.backend.set_image_data_format(FLAGS.data_format)
        _, val_set = DatasetGenerator(
            FLAGS.data_dir, FLAGS.input_size, FLAGS.input_size / 4,
            FLAGS.batch_size).run(train_num_bbox=FLAGS.train_num_bbox,
                                  val_num_bbox=FLAGS.val_num_bbox)

        for i in val_set.take(1):
            print(i)

    app.run(main)
