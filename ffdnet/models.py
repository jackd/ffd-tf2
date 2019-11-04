from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, Union, Tuple, Optional
import gin
import numpy as np
import tensorflow as tf
from kblocks import layers

from ffdnet import ops
from ffdnet.templates import FreeFormDecomposition

ArrayOrTensor = Union[tf.Tensor, np.ndarray]


@gin.configurable(module='ffdnet', blacklist=['features'])
def feature_adapter(features,
                    units=(512,),
                    use_batch_norm=True,
                    activation=tf.nn.relu6):
    for i, u in enumerate(units):
        features = layers.Dense(u, name='decoder_dense{}'.format(i))(features)
        if use_batch_norm:
            features = layers.BatchNormalization(
                name='decoder_bn{}'.format(i))(features)
        features = tf.keras.layers.Activation(
            activation, name='decoder_act{}'.format(i))(features)
    return features


@gin.configurable(module='ffdnet', blacklist=['image'])
def mobilenet2_encoder(image: tf.Tensor,
                       weights_size: int = 224,
                       alpha: float = 1.0,
                       weights: str = 'imagenet',
                       pooling: str = 'max') -> tf.Tensor:

    # create model then call with image
    # This means input_shape doesn't have to be the same as image.shape
    model = tf.keras.applications.MobileNetV2(alpha=alpha,
                                              input_shape=(weights_size,
                                                           weights_size, 3),
                                              weights=weights)
    return model(image)


def shift_probs(probs, shift: float):
    assert (probs.shape.ndims == 2 and probs.shape[1] is not None)
    T = float(probs.shape[1])
    return (1 - shift) * probs + (shift / T)


@gin.configurable(module='ffdnet', blacklist=['image', 'outputs_spec'])
def get_ffd_model(
        image: tf.Tensor,
        outputs_spec: tf.TensorSpec,  # just for cloud
        ffd_data: Tuple[FreeFormDecomposition],
        prob_shift: float = 0.1,
        encoder_fn: Callable[[tf.Tensor], tf.Tensor] = mobilenet2_encoder,
        adapter_fn: Callable[[tf.Tensor], tf.Tensor] = feature_adapter,
        prob_entropy_regularizer: Optional[
            tf.keras.regularizers.Regularizer] = None,
        deformation_regularizer: Optional[
            tf.keras.regularizers.Regularizer] = None):
    """
    Dimensions:
        B: batch size.
        T: number of templates.
        C: number of control points.
        D: number of spatial dimensions (i.e. 3 for 3D).
        N: number of points in output cloud.
        N0: number of points in template cloud, N0 >= N.

    Args:
        image: input image tensor.
        outputs_spec:
        ffd_data: (b, p) arrays/tensors
            b: [T, N0, C]
            p: [T, C, D]
        encoder_fn: function mapping image -> encoding
        adapter_fn: function mapping encoding -> pre-deform features
        entropy_penalty_factor: optional factor used in regularizing
            probabilities to encourage diversity of selection.

    Returns:
        keras Model with dict outputs:
            dp: [B, T, C, D] delta p
            probs: [B, T]
            clouds: [B, T, N, D]
    """
    features = encoder_fn(image)
    features = adapter_fn(features)
    b = []
    p = []
    for ffd in ffd_data:
        b.append(ffd.b)
        p.append(ffd.p)
    b = tf.constant(b, dtype=tf.float32)
    p = tf.constant(p, dtype=tf.float32)
    # b = tf.keras.layers.Input(tensor=b, name='b')
    # p = tf.keras.layers.Input(tensor=p, name='p')

    num_templates, num_control_points = b.shape[0], b.shape[2]
    num_dims = p.shape[-1]
    num_points = (outputs_spec.shape[-1] - 1) // num_dims

    if b.shape[1] > num_points:
        # shuffle and slice
        # shuffle only works on axis 0, hence the transposes
        b = tf.transpose(b, (1, 0, 2))
        b = tf.random.shuffle(b)
        b = b[:num_points]
        b = tf.transpose(b, (1, 0, 2))

    dp = layers.Dense(
        num_templates * num_control_points * num_dims,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-4),
        name='dp_flat')(features)
    dp = tf.keras.layers.Reshape(
        (num_templates, num_control_points, num_dims),
        name='dp',
        activity_regularizer=deformation_regularizer)(dp)
    probs = layers.Dense(num_templates,
                         activation='softmax',
                         name='pre_shift_probs')(features)
    probs = layers.Lambda(shift_probs,
                          arguments=dict(shift=prob_shift),
                          activity_regularizer=prob_entropy_regularizer,
                          name='probs')(probs)
    clouds = tf.keras.layers.Lambda(ops.deform_ffd, name='clouds')([b, p, dp])

    merged_outputs = ops.merge_outputs(clouds, probs)
    return tf.keras.Model(inputs=image, outputs=merged_outputs)
