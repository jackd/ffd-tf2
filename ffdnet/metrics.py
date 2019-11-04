from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ffdnet import ops
from tensorflow.python.keras.metrics import MeanMetricWrapper  # pylint: disable=import-error


def packed_argmax_chamfer(y_true, y_pred, num_dims=3):
    clouds, probs = ops.split_outputs(y_pred, num_dims)
    y_true, clouds, probs = ops.prepare_loss_args(y_true, y_pred, num_dims)
    return ops.argmax_chamfer(y_true, clouds, probs)


class ArgmaxChamfer(MeanMetricWrapper):

    def __init__(self, num_dims=3, name='argmax_chamfer', dtype=None):
        super(ArgmaxChamfer, self).__init__(packed_argmax_chamfer,
                                            name,
                                            dtype=dtype,
                                            num_dims=num_dims)
