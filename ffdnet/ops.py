from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
import tensorflow as tf

try:
    from pykdtree.kdtree import KDTree
except ImportError:
    from scipy.spatial import cKDTree as KDTree
    logging.warning(
        'Failed to import pykdtree, so falling back to scipy.spatial.cKDTree -'
        ' performance may suffer. Consider following instructions at '
        'https://github.com/storpipfugl/pykdtree, e.g. '
        '`conda install -c conda-forge pykdtree` ')


def check_multi_chamfer_shapes(y_true, y_pred, batch_dims=1):
    """
    Ensures batch dims are correct.

    Args:
        y_true: [..., Na, D] label points
        y_pred: [..., T, Nb, D] pred points

    Raises:
        ValueError is shapes are not consistent with the above.
    """
    if len(y_true.shape) != 2 + batch_dims:
        raise ValueError('y_true must be rank {}, got {}'.format(
            2 + batch_dims, y_true.shape))
    if len(y_pred.shape) != 3 + batch_dims:
        raise ValueError('y_pred must be rank {}, got {}'.format(
            3 + batch_dims, y_pred.shape))
    if y_true.shape[-1] != y_pred.shape[-1]:
        raise ValueError('final dims must be same, got shapes {} and {}'.format(
            y_true.shape, y_pred.shape))


def get_nearest_neighbors_np(y_true, y_pred, tree_impl=KDTree):
    """
    Get the nearest neighbor indices.

    Assumes all shape checks have been done. See `check_chamfer_single_sizes`
    for explicit checks.

    Args:
        (y_true, y_pred)
            y_true: [Na, D] label points
            y_pred: [T, Nb, D] pred points
        tree_impl: KDTree implementatione.g.
            - scipy.spatial.cKDTree
            - pykd.KDTree (fastest, but slightly tricky to install maybe?)

    Returns:
        [T, Na] ints in [0, Nb) pred indices of nearest neighbors to labels.
        [T, Nb] ints in [0, Na] label indices of nearest neighbors to preds.
    """
    T, Nb, D = y_pred.shape
    Na = y_true.shape[0]
    if y_true.shape[1] != D:
        raise ValueError('y_true')
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    tree_true = tree_impl(y_true)
    out0 = np.empty((T, Na), dtype=np.int64)
    out1 = np.empty((T, Nb), dtype=np.int64)
    for i, pred in enumerate(y_pred):
        tree_pred = tree_impl(pred)
        out0[i] = np.squeeze(tree_pred.query(y_true, 1)[1])
        out1[i] = np.squeeze(tree_true.query(pred, 1)[1])
    return out0, out1


def get_nearest_neighbors(y_true, y_pred):
    """
    Args:
        y_true: [Na, D] label points
        y_pred: [T, Nb, D] pred points

    Returns:
        [T, Na] ints in [0, Nb) pred indices of nearest neighbors to labels.
        [T, Nb] ints in [0, Na] label indices of nearest neighbors to preds.
    """
    check_multi_chamfer_shapes(y_true, y_pred, batch_dims=0)
    T, Nb = y_pred.shape[:2]
    Na = y_true.shape[0]
    pred_indices, label_indices = tf.py_function(get_nearest_neighbors_np,
                                                 (y_true, y_pred),
                                                 (tf.int64, tf.int64))
    pred_indices.set_shape((T, Na))
    label_indices.set_shape((T, Nb))
    return pred_indices, label_indices


def multi_chamfer(y_true, y_pred, mean_over_num_points=False):
    """
    Args:
        y_true: [B, Na, D] label points
        y_pred: [B, T, Nb, D] pred points
        mean_over_num_points: bool, if True take the mean rather than the sum
            over number of points.

    Returns:
        [B, T] float tensor of average chamfer losses.
    """
    Nb = y_pred.shape[2]
    Na = y_true.shape[1]
    pred_indices, label_indices = tf.map_fn(
        lambda args: get_nearest_neighbors(*args), (y_true, y_pred),
        back_prop=False,
        dtype=(tf.int64, tf.int64))
    pred_vals = tf.gather(y_pred, pred_indices, batch_dims=2)
    pred_err = tf.reduce_sum(tf.math.squared_difference(
        pred_vals, tf.expand_dims(y_true, axis=1)),
                             axis=(2, 3))
    true_vals = tf.gather(y_true, label_indices, batch_dims=1)
    label_err = tf.reduce_sum(tf.math.squared_difference(true_vals, y_pred),
                              axis=(2, 3))
    if mean_over_num_points:
        pred_err = pred_err / Na
        label_err = label_err / Nb
    return pred_err + label_err


def chamfer(y_true, y_pred, mean_over_num_points=False):
    """
    Args:
        y_true: [B, Na, D] label points
        y_pred: [B, Nb, D] pred points
        mean_over_num_points: bool, if True take the mean rather than the sum
            over number of points.

    Returns:
        [B] float tensor of average chamfer losses.
    """
    y_pred = tf.expand_dims(y_pred, axis=1)
    losses = multi_chamfer(y_true,
                           y_pred,
                           mean_over_num_points=mean_over_num_points)
    return tf.squeeze(losses, axis=1)


def argmax_chamfer(y_true, y_pred, probs_pred, mean_over_num_points=False):
    # apparently we can't gather with batch_dims == indices.shape.ndims...
    i = tf.expand_dims(tf.argmax(probs_pred, axis=-1), axis=1)
    y_pred = tf.gather(y_pred, i, batch_dims=1)
    return tf.squeeze(multi_chamfer(y_true,
                                    y_pred,
                                    mean_over_num_points=mean_over_num_points),
                      axis=1)


def deform_ffd(args):
    """
    Args:
        decomp: [T, N, C]
        control_points: [T, C, D]
        control_point_shifts: [B, T, C, D]

    Returns:
        [B, T, N, D] deformed points.
    """
    b, p, dp = args
    return tf.einsum('ijk,likm->lijm', b, p + dp)


def deform_ffd_single(args):
    """
    Args:
        decomp: [B, N, C]
        control_points: [B, C, D]
        control_point_shifts: [B, C, D]

    Returns:
        [B, N, D] deformed points
    """
    decomp, control_points, control_point_shifts = args
    return tf.matmul(decomp, control_points + control_point_shifts)


def split_outputs(merged_outputs, num_dims=3):
    """
    Args:
        merged_outputs: [B, T, N*D + 1] flattened float tensor
        num_dims: D

    Returns:
        clouds: [B, T, N, D]
        probs: [B, T]
    """
    assert (merged_outputs.shape.ndims == 3)
    num_templates, rest = merged_outputs.shape[1:]
    num_points = (rest - 1) // num_dims
    flat_clouds, probs = tf.split(merged_outputs, (-1, 1), axis=-1)
    probs = tf.squeeze(probs, axis=-1)
    clouds = tf.reshape(flat_clouds, (-1, num_templates, num_points, num_dims))
    return clouds, probs


def merge_outputs(clouds, probs):
    """
    Args:
        clouds: [B, T, N, D]
        probs: [B, T]

    Returns:
        [B, T, N*D + 1] merged, with probs last.
    """
    num_templates, num_points, num_dims = clouds.shape[1:]
    assert (probs.shape[1] == num_templates)
    clouds = tf.reshape(clouds, (-1, num_templates, num_points * num_dims))
    return tf.concat([clouds, tf.expand_dims(probs, axis=-1)], axis=-1)


def prepare_loss_args(y_true, y_pred, num_dims=3):
    clouds, probs = split_outputs(y_pred, num_dims)
    num_templates, num_points, num_dims = clouds.shape[1:]
    del num_templates
    y_true = tf.reshape(y_true, [-1, num_points, num_dims])
    return y_true, clouds, probs
