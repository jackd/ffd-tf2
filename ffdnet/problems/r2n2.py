from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf

from shape_tfds.shape.shapenet import r2n2 as sds_r2n2
from kblocks.framework.problems import TfdsProblem
from ffdnet.losses import WeightedChamfer
from ffdnet.metrics import ArgmaxChamfer


@gin.configurable(module='ffdnet')
class R2n2Problem(TfdsProblem):

    def __init__(self,
                 num_points=1024,
                 num_dims=3,
                 synset='plane',
                 loss=None,
                 metrics=None,
                 image_preprocessor=tf.keras.applications.mobilenet_v2.
                 preprocess_input):
        self._image_preprocessor = image_preprocessor
        builder = sds_r2n2.ShapenetR2n2Cloud(
            config=sds_r2n2.ShapenetR2n2CloudConfig(num_points=num_points,
                                                    synset=synset))
        if loss is None:
            loss = WeightedChamfer(num_dims=num_dims)
        if metrics is None:
            metrics = [ArgmaxChamfer(num_dims=num_dims)]
        outputs_spec = tf.TensorSpec((None, None, num_points * num_dims + 1))
        super(R2n2Problem, self).__init__(builder,
                                          loss=loss,
                                          metrics=metrics,
                                          split_map={'validation': 'test'},
                                          as_supervised=True,
                                          outputs_spec=outputs_spec)

    def _get_base_dataset(self, split):
        base = super(R2n2Problem, self)._get_base_dataset(split)

        # randomly sample an image in training or the first one otherwise
        def map_fn(renderings, labels):
            if split == 'train':
                i = tf.random.uniform((),
                                      maxval=sds_r2n2.RENDERINGS_PER_EXAMPLE,
                                      dtype=tf.int64)
            else:
                i = 0
            image = tf.cast(renderings['image'][i], tf.float32)
            image = self._image_preprocessor(image)
            return image, labels

        return base.map(map_fn, tf.data.experimental.AUTOTUNE)
