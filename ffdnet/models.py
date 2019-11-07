from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
from typing import Callable, Union, Tuple, Optional, NamedTuple, Iterable
import gin
import numpy as np
import tensorflow as tf
from kblocks.keras import applications
from kblocks.keras import layers

from ffdnet import ops
from ffdnet.templates import Decomposer


class CloudDecomposition(NamedTuple):
    decomp: np.ndarray  # [..., N, C]
    control_points: np.ndarray  # [..., C, D]


@gin.configurable(module='ffdnet')
def cloud_decompositions(decomposers: Iterable[Decomposer],
                         num_points: int = 4096) -> CloudDecomposition:
    cloud_decomps = []
    control_points = []
    for decomp in tqdm(decomposers, desc='Getting cloud decompositions'):
        cloud_decomps.append(decomp.cloud_decomp(num_points))
        control_points.append(decomp.control_points)
    return CloudDecomposition(decomp=np.array(cloud_decomps),
                              control_points=np.array(control_points))


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
def mobilenet_encoder(image: tf.Tensor,
                      weights_size: int = 224,
                      alpha: float = 1.0,
                      weights: str = 'imagenet',
                      pooling: str = 'max',
                      v2: bool = False) -> tf.Tensor:

    # create model then call with image
    # This means input_shape doesn't have to be the same as image.shape
    fn = applications.MobileNetV2 if v2 else applications.MobileNet
    model = fn(
        alpha=alpha,
        input_shape=(weights_size, weights_size, 3),
        weights=weights,
        pooling=pooling,
        include_top=False,
    )
    return model(image)


def shift_probs(probs, shift: float):
    assert (probs.shape.ndims == 2 and probs.shape[1] is not None)
    T = float(probs.shape[1])
    return (1 - shift) * probs + (shift / T)


def argmax_gather(i, *args):
    assert (i.shape.ndims == 1)
    i = tf.expand_dims(i, axis=1)
    args = tuple(
        tf.squeeze(tf.gather(a, i, batch_dims=1), axis=1) for a in args)
    if len(args) == 1:
        return args[0]
    return args


@gin.configurable(
    module='ffdnet',
    blacklist=['image', 'num_templates', 'num_control_points', 'num_dims'])
def get_deformation_params(
        image: tf.Tensor,
        num_templates: int,
        num_control_points: int,
        num_dims: int,
        prob_shift: float = 0.1,
        encoder_fn: Callable[[tf.Tensor], tf.Tensor] = mobilenet_encoder,
        adapter_fn: Callable[[tf.Tensor], tf.Tensor] = feature_adapter,
        prob_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        deformation_regularizer: Optional[
            tf.keras.regularizers.Regularizer] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Learned mapping from image to probs/deformations.

    Args:
        image: [B, H, W, 3] float batched input tensor image
        num_templates: number of templates to use
        num_control_points: number of control points to deform for each template
        prob_shift: shift applied to probabilities,
            probs = (1 - shift) * base_probs + shift / num_templates
        num_dims: number of dimensions, e.g. 3 for 3D.
        encoder_fn: function mapping image -> encoding
        adapter_fn: function mapping encoding -> pre-deform features
        prob_regularizer: activity regularizer applied to probabilities.
        deformation_regularizer: activity regularizer applied to dp.

    Returns:
        probs: [B, num_templates] probability scores
        control_point_shifts: [B, T, num_control_points, num_dims]
            control point deformations
    """
    features = encoder_fn(image)
    features = adapter_fn(features)
    control_point_shifts = layers.Dense(
        num_templates * num_control_points * num_dims,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-4),
        name='dp_flat')(features)
    control_point_shifts = tf.keras.layers.Reshape(
        (num_templates, num_control_points, num_dims),
        name='dp',
        activity_regularizer=deformation_regularizer)(control_point_shifts)
    probs = layers.Dense(num_templates,
                         activation='softmax',
                         name='pre_shift_probs')(features)
    probs = layers.Lambda(shift_probs,
                          arguments=dict(shift=prob_shift),
                          activity_regularizer=prob_regularizer,
                          name='probs')(probs)
    return probs, control_point_shifts


@gin.configurable(module='ffdnet', blacklist=['image', 'outputs_spec'])
def get_ffd_model(
        image: tf.Tensor,
        outputs_spec: Optional[tf.TensorSpec],
        cloud_decomp: Union[CloudDecomposition, Tuple[np.ndarray, np.ndarray]],
        deformation_fn=get_deformation_params,
):
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
        outputs_spec: tf.TensorSpec with shape [B, T, N * D + 1]
        cloud_decomp:
            decomp: [T, N0, C] point cloud decomposition
            control_points: [T, C, D]
        ffd_fn: function mapping
            (image, num_templates, num_control_points, num_dims) ->
            (probs, dp). See `get_deformation_params` for example.

    Returns:
        keras Model merged outputs of shape [B, T, N * D + 1]
    """
    decomp, control_points = cloud_decomp
    num_templates, n0, num_control_points = decomp.shape
    num_dims = control_points.shape[-1]

    probs, control_point_shifts = deformation_fn(image, num_templates,
                                                 num_control_points, num_dims)

    decomp = tf.convert_to_tensor(decomp, dtype=tf.float32)
    control_points = tf.convert_to_tensor(control_points, dtype=tf.float32)

    if outputs_spec is not None and outputs_spec.shape[-1] is not None:
        num_points = (outputs_spec.shape[-1] - 1) // num_dims
        if n0 > num_points:
            # shuffle and slice
            # shuffle only works on axis 0, hence the transposes
            decomp = tf.transpose(decomp, (1, 0, 2))
            decomp = tf.random.shuffle(decomp)
            decomp = decomp[:num_points]
            decomp = tf.transpose(decomp, (1, 0, 2))

    clouds = tf.keras.layers.Lambda(ops.deform_ffd, name='clouds')(
        [decomp, control_points, control_point_shifts])
    outputs = ops.merge_outputs(clouds, probs)
    model = tf.keras.Model(inputs=image, outputs=outputs)
    return model
