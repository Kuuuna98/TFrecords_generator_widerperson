import tensorflow as tf


def random_factor(minval=0, maxval=None):
    return tf.random.uniform([], minval, maxval, seed=42)


def get_factor(image):
    input_size = tf.cast(tf.shape(image)[0], tf.float32)
    last_coord = input_size - 1
    return last_coord


def random_resize(image, bboxes):
    last_coord = get_factor(image)
    random_diff = random_factor(-0.3 * last_coord, 0.3 * last_coord)
    random_diff_ratio = random_diff / last_coord
    do_a_resize_random = random_factor()

    adjusted_image = tf.cond(
        do_a_resize_random < 0.3,
        true_fn=lambda: image_resize(image, random_diff_ratio),
        false_fn=lambda: image)

    adjusted_bboxes = tf.cond(
        do_a_resize_random < 0.3,
        true_fn=lambda: bbox_resize(last_coord, bboxes, random_diff),
        false_fn=lambda: bboxes)

    return adjusted_image, adjusted_bboxes


def image_resize(image, random_diff_ratio):
    input_size = tf.cast(tf.shape(image)[0], tf.float32)
    resized_image = tf.image.crop_and_resize([image], [[
        random_diff_ratio, random_diff_ratio, 1 - random_diff_ratio,
        1 - random_diff_ratio
    ]], [0], [int(input_size), int(input_size)])[0]

    return resized_image


def bbox_resize(last_coord, bboxes, random_diff):
    valid_bboxes = bboxes[tf.reduce_any(bboxes != 0, axis=-1)]
    resized_bboxes = (valid_bboxes - random_diff) * last_coord / (
        last_coord - 2 * random_diff)

    if random_diff > 0: resized_bboxes = crop_bbox(last_coord, resized_bboxes)

    resized_bboxes = tf.pad(
        resized_bboxes,
        [[0, tf.shape(bboxes)[0] - tf.shape(resized_bboxes)[0]], [0, 0]],
        'CONSTANT')

    return resized_bboxes


def crop_bbox(last_coord, bboxes):

    valid_bboxes = bboxes[tf.reduce_all(tf.logical_and(
        tf.logical_and((bboxes[:, 0::4] < last_coord), (bboxes[:, 2::4] > 0)),
        tf.logical_and((bboxes[:, 1::4] < last_coord), (bboxes[:, 3::4] > 0))),
                                        axis=-1)]

    xmin, ymin, xmax, ymax = tf.split(value=valid_bboxes,
                                      num_or_size_splits=4,
                                      axis=1)

    xmin = tf.where(xmin < 0, 0., xmin)
    ymin = tf.where(ymin < 0, 0., ymin)
    xmax = tf.where(xmax > last_coord, last_coord, xmax)
    ymax = tf.where(ymax > last_coord, last_coord, ymax)

    cropped_bboxes = tf.concat([xmin, ymin, xmax, ymax], 1)

    return cropped_bboxes


def random_flip_lr(image, bboxes):
    do_a_flip_random = random_factor()

    adjusted_image = tf.cond(do_a_flip_random < 0.3,
                             true_fn=lambda: image_flip_lr(image),
                             false_fn=lambda: image)

    adjusted_bboxes = tf.cond(do_a_flip_random < 0.3,
                              true_fn=lambda: bbox_flip_lr(image, bboxes),
                              false_fn=lambda: bboxes)

    return adjusted_image, adjusted_bboxes


def image_flip_lr(image):
    return tf.image.flip_left_right(image)


def bbox_flip_lr(image, bboxes):
    last_coord = get_factor(image)
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
