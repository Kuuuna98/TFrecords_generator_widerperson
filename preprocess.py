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
    do_a_resize_random = random_factor() < run_criteria
    if do_a_resize_random:
        scale_factor = random_factor(scale_min, scale_max)
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
    resized_bboxes = tf.reshape(bboxes, (-1, 2))
    resized_bboxes = (resized_bboxes - center) @ [[scale_factor, 0],
                                                  [0, scale_factor]] + center
    resized_bboxes = tf.reshape(resized_bboxes, (-1, 4))
    if scale_factor > 1:
        resized_bboxes = _crop_bbox(last_coord, resized_bboxes)
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
    flipped_bboxes = tf.gather(bboxes, [2, 1, 0, 3],
                               axis=-1) * [-1., 1., -1., 1.] + [
                                   last_coord, 0., last_coord, 0.
                               ]
    return flipped_bboxes


def random_rotate(image, bboxes, run_criteria=0.3, minval=1, maxval=359):
    do_a_rotate_random = random_factor() < run_criteria
    if do_a_rotate_random:
        degree_factor = random_factor(minval, maxval)
        degrees_to_radians = math.pi / 180.0
        radians = degree_factor * degrees_to_radians
        adjusted_image = _rotate_image(image, radians)
        adjusted_bboxes = _rotate_bbox(last_coord_factor(image), bboxes,
                                       radians)
        return adjusted_image, adjusted_bboxes
    else:
        return image, bboxes


def _rotate_image(image, radians):
    return tfa.image.rotate(image, radians)


def _rotate_bbox(last_coord, bboxes, radians):
    center = last_coord / 2
    min_x, min_y, max_x, max_y = tf.split(value=bboxes,
                                          num_or_size_splits=4,
                                          axis=1)
    coordinate = tf.reshape(
        tf.concat([min_x, min_y, min_x, max_y, max_x, min_y, max_x, max_y], 1),
        (-1, 4, 2))
    rotated_coordinate = (coordinate - center) @ [[
        tf.cos(radians), -tf.sin(radians)
    ], [tf.sin(radians), tf.cos(radians)]] + center
    min_xy = tf.reshape(tf.reduce_min(rotated_coordinate, axis=1), (-1, 2))
    max_xy = tf.reshape(tf.reduce_max(rotated_coordinate, axis=1), (-1, 2))
    rotated_bboxes = tf.concat([min_xy, max_xy], 1)
    rotated_bboxes = _crop_bbox(last_coord, rotated_bboxes)
    return rotated_bboxes


def random_translateX(image, bboxes, run_criteria=0.3, minval=-8, maxval=8):
    do_a_translate_random = random_factor() < run_criteria
    if do_a_translate_random:
        translate_factor = random_factor(minval, maxval)
        adjusted_image = _translateX_image(image, translate_factor)
        adjusted_bboxes = _translateX_bbox(last_coord_factor(image), bboxes,
                                           translate_factor)
        return adjusted_image, adjusted_bboxes
    else:
        return image, bboxes


def _translateX_image(image, translateX_factor):
    return tfa.image.translate(image, [translateX_factor, 0.])


def _translateX_bbox(last_coord, bboxes, translateX_factor):
    translated_bboxes = bboxes + [translateX_factor, 0, translateX_factor, 0]
    translated_bboxes = _crop_bbox(last_coord, translated_bboxes)
    return translated_bboxes


def random_translateY(image, bboxes, run_criteria=0.3, minval=-8, maxval=8):
    do_a_translate_random = random_factor() < run_criteria
    if do_a_translate_random:
        translate_factor = random_factor(minval, maxval)
        adjusted_image = _translateY_image(image, translate_factor)
        adjusted_bboxes = _translateY_bbox(last_coord_factor(image), bboxes,
                                           translate_factor)
        return adjusted_image, adjusted_bboxes
    else:
        return image, bboxes


def _translateY_image(image, translateY_factor):
    return tfa.image.translate(image, [0., translateY_factor])


def _translateY_bbox(last_coord, bboxes, translateY_factor):
    translated_bboxes = bboxes + [0, translateY_factor, 0, translateY_factor]
    translated_bboxes = _crop_bbox(last_coord, translated_bboxes)
    return translated_bboxes


def random_shearX(image, bboxes, run_criteria=0.3, minval=-0.3, maxval=0.3):
    do_a_shear_random = random_factor() < run_criteria
    if do_a_shear_random:
        shear_factor = random_factor(minval, maxval)
        centralization_factor = shear_factor / 2 * last_coord_factor(image)
        adjusted_image = _shearX_image(image, shear_factor,
                                       centralization_factor)
        adjusted_bboxes = _shearX_bbox(last_coord_factor(image), bboxes,
                                       shear_factor, centralization_factor)
        return adjusted_image, adjusted_bboxes
    else:
        return image, bboxes


def _shearX_image(image, shearX_factor, centralizationX_factor):
    return tfa.image.transform(
        image,
        [1., shearX_factor, -1 * centralizationX_factor, 0., 1., 0., 0., 0.])


def _shearX_bbox(last_coord, bboxes, shearX_factor, centralizationX_factor):
    min_x, min_y, max_x, max_y = tf.split(value=bboxes,
                                          num_or_size_splits=4,
                                          axis=1)
    sheared_coordinate = tf.reshape(
        tf.concat([min_x, min_y, min_x, max_y, max_x, min_y, max_x, max_y], 1),
        (-1, 4, 2))
    sheared_coordinate = (sheared_coordinate) @ [[1., 0.],
                                                 [-1 * shearX_factor, 1.]] + [
                                                     centralizationX_factor, 0.
                                                 ]
    min_xy = tf.reshape(tf.reduce_min(sheared_coordinate[:], axis=1), (-1, 2))
    max_xy = tf.reshape(tf.reduce_max(sheared_coordinate[:], axis=1), (-1, 2))
    sheared_bboxes = tf.concat([min_xy, max_xy], 1)
    sheared_bboxes = _crop_bbox(last_coord, sheared_bboxes)
    return sheared_bboxes


def random_shearY(image, bboxes, run_criteria=0.3, minval=-0.3, maxval=0.3):
    do_a_shear_random = random_factor() < run_criteria
    if do_a_shear_random:
        shear_factor = random_factor(minval, maxval)
        centralization_factor = shear_factor / 2 * last_coord_factor(image)
        adjusted_image = _shearY_image(image, shear_factor,
                                       centralization_factor)
        adjusted_bboxes = _shearY_bbox(last_coord_factor(image), bboxes,
                                       shear_factor, centralization_factor)
        return adjusted_image, adjusted_bboxes
    else:
        return image, bboxes


def _shearY_image(image, shearY_factor, centralizationY_factor):
    return tfa.image.transform(
        image,
        [1., 0., 0., shearY_factor, 1., -1 * centralizationY_factor, 0., 0.])


def _shearY_bbox(last_coord, bboxes, shearY_factor, centralizationY_factor):
    min_x, min_y, max_x, max_y = tf.split(value=bboxes,
                                          num_or_size_splits=4,
                                          axis=1)
    sheared_coordinate = tf.reshape(
        tf.concat([min_x, min_y, min_x, max_y, max_x, min_y, max_x, max_y], 1),
        (-1, 4, 2))
    sheared_coordinate = (sheared_coordinate) @ [[1., -1 * shearY_factor],
                                                 [0., 1.]] + [
                                                     0., centralizationY_factor
                                                 ]
    min_xy = tf.reshape(tf.reduce_min(sheared_coordinate[:], axis=1), (-1, 2))
    max_xy = tf.reshape(tf.reduce_max(sheared_coordinate[:], axis=1), (-1, 2))
    sheared_bboxes = tf.concat([min_xy, max_xy], 1)
    sheared_bboxes = _crop_bbox(last_coord, sheared_bboxes)
    return sheared_bboxes
