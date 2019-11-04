from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gin


@gin.configurable(module='ffdnet')
def get_model_dir(root_dir, subdir, arch_id=None, synset_id=None):
    args = (k for k in (root_dir, subdir, arch_id, synset_id) if k is not None)
    return os.path.expanduser(os.path.expandvars(os.path.join(*args)))


@gin.configurable(module='ffdnet')
def get_synset_id(synset='telephone'):
    from shape_tfds.shape.shapenet.r2n2 import synset_id
    return synset_id(synset)
