from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import trimesh

import numpy as np
import tensorflow as tf
import gin

from typing import Iterable, Optional, NamedTuple, Union, Sequence
from tqdm import tqdm

from kblocks.callbacks import CheckpointCallback
from kblocks.framework.problems import Problem
from ffdnet.models import get_ffd_model
from ffdnet.templates import Decomposer
from ffdnet import ops


class MeshDecomposition(NamedTuple):
    cloud_decomp: np.ndarray
    vertex_decomp: Union[np.ndarray, Sequence[np.ndarray]]
    faces: Union[np.ndarray, Sequence[np.ndarray]]
    control_points: np.ndarray


@gin.configurable(module='ffdnet')
def cloud_mesh_decompositions(
        decomposers: Iterable[Decomposer],
        num_points: int = 4096,
        max_len: Optional[float] = 0.05,
) -> MeshDecomposition:
    cloud_decomps = []
    vertex_decomps = []
    faces = []
    control_points = []
    for decomp in tqdm(decomposers, desc='Getting mesh decompositions'):
        cloud_decomps.append(decomp.cloud_decomp(num_points))
        vertex_decomps.append(decomp.vertex_decomp(max_len))
        faces.append(decomp.faces(max_len))
        control_points.append(decomp.control_points)
    return MeshDecomposition(
        np.array(cloud_decomps),
        tuple(vertex_decomps),
        tuple(faces),
        np.array(control_points),
    )


def restore(model: tf.keras.Model, model_dir: str) -> int:
    import os
    cb = CheckpointCallback(os.path.join(model_dir, 'chkpts'))
    cb.set_model(model)
    ckpt = cb.checkpoint()
    cb.restore(ckpt).expect_partial()  # no optimizer params loaded
    return cb.epoch(ckpt)


def pad_arrays(arrs):
    lens = tuple(len(arr) for arr in arrs)
    max_len = max(lens)
    out = np.zeros((len(arrs), max_len) + arrs[0].shape[1:],
                   dtype=arrs[0].dtype)
    for i, arr in enumerate(arrs):
        out[i, :len(arr)] = arr
    return out, lens


def broadcast_colors(color, size):
    out = np.empty((size, 4), dtype=np.uint8)
    if len(color) == 3:
        out[:, :3] = color
        out[:, 3] = 255
    else:
        out[:] = color
    return out


def point_cloud(vertices, color=None):
    colors = None if color is None else broadcast_colors(
        color, vertices.shape[0])
    return trimesh.PointCloud(vertices, colors=colors)


def argmax_gather(i, *args):
    assert (i.shape.ndims == 1)
    i = tf.expand_dims(i, axis=1)
    args = tuple(
        tf.squeeze(tf.gather(a, i, batch_dims=1), axis=1) for a in args)
    if len(args) == 1:
        return args[0]
    return args


@gin.configurable(module='ffdnet')
class Deformer(object):

    def __init__(self,
                 model_dir,
                 mesh_decomps: MeshDecomposition,
                 problem: Problem,
                 image_spec=tf.TensorSpec(shape=(None, None, 3),
                                          dtype=tf.float32),
                 model_fn=get_ffd_model):
        self.problem = problem
        self.faces = mesh_decomps.faces
        control_points = mesh_decomps.control_points
        vertex_decomps, self.v_lens = pad_arrays(mesh_decomps.vertex_decomp)
        cloud_decomps = mesh_decomps.cloud_decomp
        self.n0 = cloud_decomps.shape[1]
        decomps = np.concatenate([cloud_decomps, vertex_decomps], axis=1)
        ffd_decomp = (decomps, control_points)
        image = tf.keras.layers.Input(shape=image_spec.shape,
                                      dtype=image_spec.dtype)
        outputs_spec = None
        base_model = model_fn(image, outputs_spec, ffd_decomp)
        epoch = restore(base_model, model_dir)
        logging.info('Restored model parameters from epoch {}'.format(epoch))
        clouds, probs = ops.split_outputs(base_model.outputs[0],
                                          num_dims=control_points.shape[-1])
        index = tf.argmax(probs, axis=-1)
        cloud = argmax_gather(index, clouds)
        self.model = tf.keras.Model(inputs=image, outputs=(cloud, index))

    def __call__(self, image):
        assert (image.shape.ndims == 3)
        image = tf.expand_dims(image, axis=0)
        cloud, index = self.model(image)
        i = tf.squeeze(index, axis=0).numpy()
        points = tf.squeeze(cloud, axis=0)
        cloud = points[:self.n0].numpy()
        vertices = points[self.n0:self.n0 + self.v_lens[i]].numpy()
        faces = self.faces[i]

        return vertices, faces, cloud

    def vis_all(self, split, shuffle=False, learning_phase=0):
        import matplotlib.pyplot as plt
        tf.keras.backend.set_learning_phase(learning_phase)
        dataset = self.problem.get_base_dataset(split)
        if shuffle:
            dataset = dataset.shuffle(self.problem.shuffle_buffer)
        for image, label in dataset:
            label = label.numpy()
            vertices, faces, cloud = self(image)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            scene = mesh.scene()
            scene.add_geometry(point_cloud(cloud, [0, 0, 255]))
            scene.add_geometry(point_cloud(label, [0, 255, 0]))
            image = image.numpy()
            image -= np.min(image)
            image /= np.max(image)
            plt.imshow(image)
            scene.show(background=(0, 0, 0, 0))


@gin.configurable(module='ffdnet')
def vis_all(deformer, split, shuffle=False, learning_phase=0):
    deformer.vis_all(split, shuffle, learning_phase)
