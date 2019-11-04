from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
from kblocks.optimizers import scope as opt_scope


def _annealed_factor(anneal_steps):
    return tf.exp(-tf.cast(opt_scope.get_iterations(), tf.float32) /
                  anneal_steps)


@gin.configurable(module='ffdnet')
class AnnealedRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, base, anneal_steps):
        """
        Args:
            base: unannealed tf.keras.regularizers.Regularizer or callable
                instance.
            anneal_steps: number of optimizer steps for value to reduce by
                a factor of e.
        """
        self.base = tf.keras.regularizers.get(base)
        self.anneal_steps = anneal_steps

    def get_config(self):
        return dict(base=tf.keras.util.serialize_keras_object(self.base),
                    anneal_steps=self.anneal_steps)

    def call(self, x):
        return _annealed_factor(self.anneal_steps) * self.base(x)


@gin.configurable(module='ffdnet')
class EntropyRegularizer(tf.keras.regularizers.Regularizer):
    """
    Regularizer that encourages diversity inspired by entropy loss.

    ```python
    f = factor if anneal_steps is None else factor * tf.exp(-t / anneal_steps)
    loss = f * x * log(x)
    ```

    t is the number of optimizer steps.
    """

    def __init__(self, factor):
        """
        Args:
            factor: initial scaling factor.
        """
        self.factor = factor

    def get_config(self):
        return dict(factor=tf.keras.utils.serialize_keras_object(self.factor))

    def __call__(self, x: tf.Tensor):
        if x.shape.ndims != 2:
            raise ValueError(
                'Expected `x` to be rank-2, got tensor with shape {}'.format(
                    x.shape))

        return self.factor * tf.reduce_sum(x * tf.math.log(x), axis=-1)
