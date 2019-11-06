from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import trimesh
from shape_tfds.shape.shapenet.r2n2 import synset_id
from ffdnet.vis import cloud_mesh_decompositions
from ffdnet.templates import get_default_decomposers

sid = synset_id('plane')
decomposers = get_default_decomposers(synset_id('plane'))
cloud_decomps, vertex_decomps, faces, control_points = \
        cloud_mesh_decompositions(decomposers)

for d, d0, c in zip(cloud_decomps, vertex_decomps, control_points):
    colors0 = np.zeros((d0.shape[0], 4), dtype=np.uint8)
    colors0[:] = [0, 255, 0, 255]
    scene = trimesh.PointCloud(np.matmul(d0, c), colors=colors0).scene()
    # scene.add_geometry(trimesh.Trimesh(vertices=np.matmul(d0, c), faces=f))
    colors = np.zeros((d.shape[0], 4), dtype=np.uint8)
    colors[:] = [255, 0, 0, 255]
    scene.add_geometry(
        trimesh.PointCloud(vertices=np.matmul(d, c), colors=colors))
    scene.show(background=(0, 0, 0))
