import tensorflow as tf


def random_factor(minval=0, maxval=None):
    return tf.random.uniform([], minval, maxval, seed=42)


def last_coord_factor(image):
    input_size = tf.cast(tf.shape(image)[0], tf.float32)
    last_coord = input_size - 1
    return last_coord


def random_resize(image,
                  bboxes,
                  run_criteria=0.3,
                  scale_min=0.7,
                  scale_max=1.3):
    scale_factor = random_factor(scale_min, scale_max)
    do_a_resize_random = random_factor() < run_criteria

    if do_a_resize_random:
        adjusted_image = _resize_image(image, scale_factor)
        adjusted_bboxes = _resize_bbox(last_coord_factor(image), bboxes,
                                       scale_factor)
        return adjusted_image, adjusted_bboxes
    else:
        return image, bboxes


def _resize_image(image, scale_factor):
    input_size = tf.cast(tf.shape(image)[0], tf.float32)
    box_indices = 0.5 + 0.5 / tf.convert_to_tensor(
        [[-scale_factor, -scale_factor, scale_factor, scale_factor]])
    resized_image = tf.image.crop_and_resize(
        [image], box_indices, [0],
        [int(input_size), int(input_size)])[0]
    return resized_image


def _resize_bbox(last_coord, bboxes, scale_factor):
    center = last_coord / 2
    valid_bboxes = bboxes[tf.reduce_any(bboxes != 0, axis=-1)]
    valid_bboxes = tf.reshape(valid_bboxes, (-1, 2))
    resized_bboxes = (valid_bboxes - center) @ [[scale_factor, 0],
                                                [0, scale_factor]] + center
    resized_bboxes = tf.reshape(resized_bboxes, (-1, 4))

    if scale_factor > 1:
        resized_bboxes = _crop_bbox(last_coord, resized_bboxes)

    resized_bboxes = tf.pad(
        resized_bboxes,
        [[0, tf.shape(bboxes)[0] - tf.shape(resized_bboxes)[0]], [0, 0]],
        'CONSTANT')
    return resized_bboxes


def _crop_bbox(last_coord, bboxes):
    clipped_bboxes = tf.clip_by_value(bboxes,
                                      clip_value_min=0,
                                      clip_value_max=last_coord)
    matrix_bboxes = tf.reshape(clipped_bboxes, (-1, 2, 2))
    bboxes_length = [[[-1., 1.]]] @ matrix_bboxes
    clipped_bboxes = clipped_bboxes[tf.squeeze(tf.reduce_all(
        bboxes_length != 0, axis=-1),
                                               axis=-1)]

    return clipped_bboxes


def random_flip_lr(image, bboxes, run_criteria=0.3):
    do_a_flip_random = random_factor() < run_criteria

    if do_a_flip_random:
        adjusted_image = _flip_lr_image(image)
        adjusted_bboxes = _flip_lr_bbox(last_coord_factor(image), bboxes)
        return adjusted_image, adjusted_bboxes
    else:
        return image, bboxes


def _flip_lr_image(image):
    return tf.image.flip_left_right(image)


def _flip_lr_bbox(last_coord, bboxes):
    valid_bboxes = bboxes[tf.reduce_any(bboxes != 0, axis=-1)]
    flipped_bboxes = tf.gather(valid_bboxes, [2, 1, 0, 3],
                               axis=-1) * [-1., 1., -1., 1.] + [
                                   last_coord, 0., last_coord, 0.
                               ]
    flipped_bboxes = tf.pad(
        flipped_bboxes,
        [[0, tf.shape(bboxes)[0] - tf.shape(flipped_bboxes)[0]], [0, 0]],
        'CONSTANT')

    return flipped_bboxes
