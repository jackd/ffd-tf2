from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
from ffdnet import ops


class Weighting(object):
    LINEAR = 'linear'
    ENTROPY = 'entropy'

    @classmethod
    def all(cls):
        return (Weighting.LINEAR, Weighting.ENTROPY)

    @classmethod
    def validate(cls, x):
        if x not in cls.all():
            raise ValueError('Invalid weighting {} - must be in {}'.format(
                x, cls.all()))


_weight_fns = {
    Weighting.LINEAR: lambda x: x,
    Weighting.ENTROPY: lambda x: -tf.math.log(1 - x)
}


def prob_weights(probs: tf.Tensor,
                 weighting: str = Weighting.LINEAR) -> tf.Tensor:
    Weighting.validate(weighting)
    return _weight_fns[weighting](probs)


@gin.configurable(module='ffdnet')
class WeightedChamfer(tf.keras.losses.Loss):

    def __init__(self,
                 num_templates: int = 30,
                 num_dims: int = 3,
                 weighting: str = Weighting.LINEAR,
                 **kwargs):
        Weighting.validate(weighting)
        self.num_dims = num_dims
        self.num_templates = num_templates
        self.weighting = weighting
        super(WeightedChamfer, self).__init__(**kwargs)

    def get_config(self):
        return dict(num_dims=self.num_dims,
                    num_templates=self.num_templates,
                    weighting=self.weighting)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: reshaped to [B, N, 3], point cloud labels
            y_pred: [B, T * (C * D + 1)] flattened combined clouds and probs

        Returns:
            [B] float tensor.
        """
        y_true, clouds, probs = ops.prepare_loss_args(y_true, y_pred,
                                                      self.num_dims)
        losses = ops.multi_chamfer(y_true, clouds)
        return tf.reduce_sum(prob_weights(probs, self.weighting) * losses,
                             axis=-1)
