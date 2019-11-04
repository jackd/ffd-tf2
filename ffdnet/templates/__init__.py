from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import os
import numpy as np
import h5py
from typing import NamedTuple, Optional, Tuple


class FreeFormDecomposition(NamedTuple):
    b: np.ndarray  # [N, C]
    p: np.ndarray  # [C, D]
    faces: Optional[np.ndarray]  # [F, 3] values in [0, N)


@gin.configurable(module='ffdnet')
def get_default_template_ids(synset_id: str):
    path = os.path.join(os.path.dirname(__file__), 'default_ids',
                        '{}.txt'.format(synset_id))
    if not os.path.isfile(path):
        raise IOError('No template_ids file found at {}'.format(path))
    return np.loadtxt(path, dtype=str)


@gin.configurable(module='ffdnet')
def get_default_decompositions_file(
        synset_id: str,
        num_dims: int = 3,
        num_points: int = 4096,
        cache_dir: str = '~/ffdnet',
) -> h5py.File:
    from shape_tfds.shape.shapenet.core.base import cloud_loader_context
    from ffd import deform
    path = os.path.expanduser(
        os.path.expandvars(
            os.path.join(cache_dir, 'templates',
                         'd{}-p{}'.format(num_dims, num_points),
                         '{}.h5'.format(synset_id))))
    if not os.path.isfile(path):
        from shape_tfds.shape.shapenet.core.base import cloud_loader_context
        folder = os.path.dirname(path)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        loader_context = cloud_loader_context(synset_id, num_points=num_points)
        model_ids = get_default_template_ids(synset_id)
        try:
            with h5py.File(path, 'w') as fp:
                from tqdm import tqdm
                with loader_context as loader:
                    for model_id in tqdm(model_ids,
                                         desc='Creating FFD d{}-p{}/{}'.format(
                                             num_dims, num_points, synset_id)):
                        points = loader[model_id]
                        b, p = deform.get_ffd(points, dims=num_dims)
                        model_group = fp.create_group(model_id)
                        for k, v in (('b', b), ('p', p), ('points', points)):
                            model_group.create_dataset(k, data=v)
        except (Exception, KeyboardInterrupt):
            if os.path.isfile(path):
                os.remove(path)
            raise
    return h5py.File(path, 'r')


@gin.configurable(module='ffdnet')
def get_default_decompositions(synset_id: str,
                               dims: int = 3,
                               num_points: int = 4096,
                               cache_dir: str = '~/ffdnet'
                              ) -> Tuple[FreeFormDecomposition, ...]:
    decomps = []
    with get_default_decompositions_file(synset_id, dims, num_points,
                                         cache_dir) as fp:
        for model_id in sorted(fp):
            g = fp[model_id]
            decomps.append(
                FreeFormDecomposition(np.array(g['b']), np.array(g['p']), None))
    return tuple(decomps)


# from shape_tfds.shape.shapenet.core.base import zipped_mesh_loader_context

# with zipped_mesh_loader_context('04401088') as loader:
#     keys = sorted(loader.keys())
#     keys = keys[:int(0.8 * len(keys))]
#     keys = np.random.choice(keys, 30, replace=False)
#     print('\n'.join(keys))

if __name__ == '__main__':
    from shape_tfds.shape.shapenet.r2n2 import synset_ids
    for synset_id in synset_ids():
        get_default_decompositions(synset_id)
    # from shape_tfds.shape.shapenet.r2n2 import synset_id
    # get_default_decompositions(synset_id('telephone'))
