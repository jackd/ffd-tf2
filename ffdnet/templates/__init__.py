from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from tqdm import tqdm
import gin
import os
import numpy as np
import h5py
from typing import NamedTuple, Tuple, Optional, Iterable
import collections

from ffd import deform


class Mesh(NamedTuple):
    vertices: np.ndarray
    faces: np.ndarray


def _load_mesh(group: h5py.Group) -> Mesh:
    return Mesh(group['vertices'][:], group['faces'][:])


class Decomposer(object):
    """Class for computing/caching/loading decompositions."""

    def __init__(self, root: h5py.Group):
        self._root = root

    @staticmethod
    def create(root: h5py.Group, vertices: np.ndarray, faces: np.ndarray,
               control_dims: Tuple[int, int, int]) -> 'Decomposer':
        if not all(c == control_dims[0] for c in control_dims[1:]):
            raise NotImplementedError(
                'all control_dims must be same at the moment')
        vertices = vertices.astype(np.float32)
        faces = faces.astype(np.int64)
        root.attrs['control_dims'] = control_dims
        stu_origin, stu_axes = deform.get_stu_params(vertices)
        root.attrs['stu_origin'] = stu_origin
        root.attrs['stu_axes'] = stu_axes
        root.create_dataset('vertices', data=vertices.astype(np.float32))
        root.create_dataset('faces', data=faces.astype(np.int64))
        root.create_dataset('control_points',
                            data=deform.get_control_points(
                                control_dims[0], stu_origin, stu_axes))
        root.create_group('clouds')
        root.create_group('subdivs')
        return Decomposer(root)

    @property
    def control_points(self):
        return self._root['control_points'][:]

    @property
    def control_dims(self):
        return self._root.attrs['control_dims'][:]

    @property
    def stu_origin(self):
        return self._root.attrs['stu_origin'][:]

    @property
    def stu_axes(self):
        return self._root.attrs['stu_axes'][:]

    def _mesh_group(self, max_len: Optional[float] = None) -> h5py.File:
        if max_len is None:
            return self._root
        else:
            return self._root.require_group['s{:d}'.format(int(1000 * max_len))]

    def _mesh(self, max_len: Optional[float] = None) -> h5py.Group:
        import trimesh
        if max_len is None:
            return self._root
        subdivs = self._root['subdivs']
        key = 's{:d}'.format(int(1000 * max_len))
        if key in subdivs:
            return subdivs[key]
        group = self._root
        # TODO: find cached subdivided mesh with smallest larger max_len
        vertices, faces = _load_mesh(group)
        vertices, faces = trimesh.remesh.subdivide_to_size(vertices,
                                                           faces,
                                                           max_len,
                                                           max_iter=1000)
        vertices = vertices.astype(np.float32)
        group = subdivs.create_group(key)
        group.attrs['max_len'] = max_len
        group.create_dataset('vertices', data=vertices)
        group.create_dataset('faces', data=faces)
        return group

    def mesh(self, max_len: Optional[float] = None) -> Mesh:
        return _load_mesh(self._mesh(max_len))

    def faces(self, max_len: Optional[float] = None) -> np.ndarray:
        return self.mesh(max_len).faces

    def _cloud(self, num_points: int) -> h5py.Group:
        from shape_tfds.shape.shapenet.core.base import sample_faces
        group = self._root['clouds'].require_group('n{}'.format(num_points))
        if 'vertices' not in group:
            vertices, faces = _load_mesh(self._root)
            cloud = sample_faces(vertices, faces, num_points).astype(np.float32)
            group.create_dataset('vertices', data=cloud)
        return group

    def cloud(self, num_points: int) -> np.ndarray:
        return self._cloud(num_points)['vertices'][:]

    def _decomp(self, group) -> np.ndarray:
        if 'decomp' in group:
            return group['decomp'][:]
        vertices = group['vertices'][:]
        decomp = deform.get_deformation_matrix(vertices, self.control_dims[0],
                                               self.stu_origin, self.stu_axes)
        group.create_dataset('decomp', data=decomp)
        return decomp

    def cloud_decomp(self, num_points: int) -> np.ndarray:
        return self._decomp(self._cloud(num_points))

    def vertex_decomp(self, max_len: Optional[float]) -> np.ndarray:
        return self._decomp(self._mesh(max_len))


@gin.configurable(module='ffdnet')
class DecomposerSequence(collections.Sequence):

    def __init__(self, path: str, model_ids: Iterable[str], mode: str = 'a'):
        if not os.path.isfile(path):
            raise ValueError('No file at path {}'.format(path))
        self._path = path
        self._model_ids = tuple(model_ids)
        self._fp = h5py.File(self._path, mode=mode)
        a, b, c = self._fp.attrs['control_dims']  # pylint: disable=unpacking-non-sequence
        self._control_dims = (a, b, c)  # make pyright play nicely...
        self._synset_id: str = self._fp.attrs['synset_id']

    @property
    def path(self) -> str:
        return self._path

    @property
    def model_ids(self) -> Tuple[str, ...]:
        return self._model_ids

    @staticmethod
    def create(
            path: str,
            synset_id: str,
            model_ids=Iterable[str],
            control_dims: Tuple[int, int, int] = (3, 3, 3),
    ) -> 'DecomposerSequence':
        if os.path.isfile(path):
            logging.info('File already exists at {} - '
                         'skipping decomposer creation'.format(path))
        else:
            from shape_tfds.shape.shapenet.core.base import \
                zipped_mesh_loader_context
            import trimesh
            logging.info('Creating decomposers at {}'.format(path))

            folder = os.path.dirname(path)
            if not os.path.isdir(folder):
                os.makedirs(folder)

            def map_fn(key, value):
                if not isinstance(value, trimesh.Trimesh):
                    meshes = [
                        trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                        for m in value.geometry.values()
                    ]
                    value = trimesh.util.concatenate(meshes)
                return value.vertices, value.faces

            loader_context = zipped_mesh_loader_context(synset_id,
                                                        item_map_fn=map_fn)
            try:
                with h5py.File(path, 'a') as fp:
                    fp.attrs['synset_id'] = synset_id
                    fp.attrs['control_dims'] = control_dims
                    with loader_context as loader:
                        for model_id in tqdm(
                                model_ids, desc='Creating FFD {}'.format(path)):
                            model_group = fp.create_group(model_id)
                            vertices, faces = loader[model_id]
                            Decomposer.create(model_group,
                                              vertices,
                                              faces,
                                              control_dims=control_dims)

            except (Exception, KeyboardInterrupt):
                if os.path.isfile(path):
                    os.remove(path)
                raise
        return DecomposerSequence(path, model_ids)

    @property
    def control_dims(self) -> Tuple[int, int, int]:
        return self._control_dims

    @property
    def synset_id(self) -> str:
        return self._synset_id

    def __len__(self):
        return len(self.model_ids)

    def __iter__(self):

        def f():
            for model_id in self.model_ids:
                yield Decomposer(self._fp[model_id])

        return iter(f())

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [Decomposer(self._model_ids[ii]) for ii in i]
        else:
            return Decomposer(self._model_ids[i])

    def close(self):
        self._fp.close()


@gin.configurable(module='ffdnet')
def get_default_template_ids(synset_id: str):
    path = os.path.join(os.path.dirname(__file__), 'default_ids',
                        '{}.txt'.format(synset_id))
    if not os.path.isfile(path):
        raise IOError('No template_ids file found at {}'.format(path))
    return np.loadtxt(path, dtype=str)


@gin.configurable(module='ffdnet')
def get_default_decomposers(synset_id: str,
                            control_dims: Tuple[int, int, int] = (3, 3, 3),
                            cache_dir: str = '~/ffdnet') -> DecomposerSequence:
    folder = os.path.expanduser(
        os.path.expandvars(
            os.path.join(cache_dir, 'templates',
                         'c{}'.format('-'.join(str(c) for c in control_dims)))))
    if not os.path.isdir(folder):
        os.makedirs(folder)
    path = os.path.join(folder, '{}.h5'.format(synset_id))
    model_ids = get_default_template_ids(synset_id)
    return DecomposerSequence.create(path, synset_id, model_ids, control_dims)
