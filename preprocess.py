import tensorflow as tf
import tensorflow_addons as tfa
import math


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
        adjusted_bboxes = _resize_bbox(bboxes, last_coord_factor(image),
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


def _resize_bbox(bboxes, last_coord, scale_factor):
    center = last_coord / 2
    valid_bboxes = bboxes[tf.reduce_any(bboxes != 0, axis=-1)]
    valid_bboxes = tf.reshape(valid_bboxes, (-1, 2))
    resized_bboxes = (valid_bboxes - center) @ [[scale_factor, 0],
                                                [0, scale_factor]] + center
    resized_bboxes = tf.reshape(resized_bboxes, (-1, 4))

    if scale_factor > 1:
        resized_bboxes = _crop_bbox(resized_bboxes, last_coord)

    resized_bboxes = tf.pad(
        resized_bboxes,
        [[0, tf.shape(bboxes)[0] - tf.shape(resized_bboxes)[0]], [0, 0]],
        'CONSTANT')
    return resized_bboxes


def _crop_bbox(bboxes, last_coord):
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
        adjusted_bboxes = _flip_lr_bbox(bboxes, last_coord_factor(image))
        return adjusted_image, adjusted_bboxes
    else:
        return image, bboxes


def _flip_lr_image(image):
    return tf.image.flip_left_right(image)


def _flip_lr_bbox(bboxes, last_coord):
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


def random_rotate(image, bboxes, run_criteria=0.3, minval=1, maxval=359):
    do_a_rotate_random = random_factor() < run_criteria
    degree_factor = random_factor(120, 120)
    degrees_to_radians = math.pi / 180.0
    radians = degree_factor * degrees_to_radians

    if do_a_rotate_random:
        adjusted_image = _rotate_image(image, radians)
        adjusted_bboxes = _rotate_bbox(bboxes, last_coord_factor(image),
                                       radians, degree_factor)
        return adjusted_image, adjusted_bboxes
    else:
        return image, bboxes


def _rotate_image(image, radians):
    rotated_image = tfa.image.rotate(image, radians)

    return rotated_image


def _rotate_bbox(bboxes, last_coord, radians, degree_factor):
    center = last_coord / 2
    valid_bboxes = bboxes[tf.reduce_any(bboxes != 0, axis=-1)]
    factor = radians

    if (90 <= degree_factor < 180) or (270 <= degree_factor < 360):
        valid_bboxes = tf.gather(valid_bboxes, [2, 1, 0, 3], axis=-1)

    valid_bboxes_x = tf.reshape(valid_bboxes, (-1, 2))
    rotated_bboxes_x = (valid_bboxes_x - center) @ [[
        tf.cos(factor), -tf.sin(factor)
    ], [tf.sin(factor), tf.cos(factor)]] + center

    valid_bboxes_y = tf.gather(valid_bboxes, [2, 1, 0, 3], axis=-1)
    valid_bboxes_y = tf.reshape(valid_bboxes_y, (-1, 2))
    rotated_bboxes_y = (valid_bboxes_y - center) @ [[
        tf.cos(factor), -tf.sin(factor)
    ], [tf.sin(factor), tf.cos(factor)]] + center

    x, _ = tf.split(value=rotated_bboxes_x, num_or_size_splits=2, axis=1)
    _, y = tf.split(value=rotated_bboxes_y, num_or_size_splits=2, axis=1)

    x = tf.reshape(x, (-1, 2))
    y = tf.reshape(y, (-1, 2))

    x_min = tf.reshape(tf.reduce_min(x[:], axis=-1), (-1, 1))
    x_max = tf.reshape(tf.reduce_max(x[:], axis=-1), (-1, 1))
    y_min = tf.reshape(tf.reduce_min(y[:], axis=-1), (-1, 1))
    y_max = tf.reshape(tf.reduce_max(y[:], axis=-1), (-1, 1))

    rotated_bboxes = tf.concat([x_min, y_min, x_max, y_max], 1)

    rotated_bboxes = _crop_bbox(last_coord, rotated_bboxes)

    rotated_bboxes = tf.pad(
        rotated_bboxes,
        [[0, tf.shape(bboxes)[0] - tf.shape(rotated_bboxes)[0]], [0, 0]],
        'CONSTANT')

    return rotated_bboxes


def random_translateX(image,
                      bboxes,
                      run_criteria=0.3,
                      minval=-255.5,
                      maxval=255.5):
    do_a_translate_random = random_factor() < run_criteria
    translate_factor = random_factor(minval, maxval)

    if do_a_translate_random:
        adjusted_image = _translateX_image(image, translate_factor)
        adjusted_bboxes = _translateX_bbox(last_coord_factor(image), bboxes,
                                           translate_factor)
        return adjusted_image, adjusted_bboxes
    else:
        return image, bboxes


def _translateX_image(image, translateX_factor):
    translated_image = tfa.image.translate(image, [translateX_factor, 0.])

    return translated_image


def _translateX_bbox(last_coord, bboxes, translateX_factor):
    valid_bboxes = bboxes[tf.reduce_any(bboxes != 0, axis=-1)]
    translated_bboxes = valid_bboxes + [
        translateX_factor, 0, translateX_factor, 0
    ]
    translated_bboxes = _crop_bbox(last_coord, translated_bboxes)

    translated_bboxes = tf.pad(
        translated_bboxes,
        [[0, tf.shape(bboxes)[0] - tf.shape(translated_bboxes)[0]], [0, 0]],
        'CONSTANT')
    return translated_bboxes


def random_translateY(image,
                      bboxes,
                      run_criteria=0.3,
                      minval=-255.5,
                      maxval=255.5):
    do_a_translate_random = random_factor() < run_criteria
    translate_factor = random_factor(minval, maxval)

    if do_a_translate_random:
        adjusted_image = _translateY_image(image, translate_factor)
        adjusted_bboxes = _translateY_bbox(last_coord_factor(image), bboxes,
                                           translate_factor)
        return adjusted_image, adjusted_bboxes
    else:
        return image, bboxes


def _translateY_image(image, translateY_factor):
    translated_image = tfa.image.translate(image, [0., translateY_factor])
    return translated_image


def _translateY_bbox(last_coord, bboxes, translateY_factor):

    valid_bboxes = bboxes[tf.reduce_any(bboxes != 0, axis=-1)]
    translated_bboxes = valid_bboxes + [
        0, translateY_factor, 0, translateY_factor
    ]
    translated_bboxes = _crop_bbox(last_coord, translated_bboxes)

    translated_bboxes = tf.pad(
        translated_bboxes,
        [[0, tf.shape(bboxes)[0] - tf.shape(translated_bboxes)[0]], [0, 0]],
        'CONSTANT')
    return translated_bboxes
