from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from ffdnet import ops

y_true = np.array([[0.], [1.]], dtype=np.float32)
y_pred = np.array([
    [
        [0.1],
        [0.9],
        [0.2],
    ],
    [
        [0.8],
        [0.6],
        [-0.1],
    ],
],
                  dtype=np.float32)
expected_pred_indices = np.array([[0, 1], [2, 0]], dtype=np.int64)
expected_true_indices = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int64)
expected_dist0 = np.array([[0.1, 0.1, 0.2], [0.2, 0.4, 0.1]], dtype=np.float32)
expected_dist1 = np.array([[0.1, 0.1], [0.1, 0.2]], dtype=np.float32)
expected_chamfer = np.mean(expected_dist0**2, axis=-1) + \
        np.mean(expected_dist1**2, axis=-1)


class OpsTest(tf.test.TestCase):

    def test_nearest_neighbors_np(self):
        pred_indices, true_indices = ops.get_nearest_neighbors_np(
            y_true, y_pred)
        np.testing.assert_equal(pred_indices, expected_pred_indices)
        np.testing.assert_equal(true_indices, expected_true_indices)

    def test_nearest_neighbors(self):
        pred_indices, true_indices = self.evaluate(
            ops.get_nearest_neighbors(y_true, y_pred))
        np.testing.assert_equal(pred_indices, expected_pred_indices)
        np.testing.assert_equal(true_indices, expected_true_indices)

    def test_multi_chamfer(self):
        b_true = np.expand_dims(y_true, axis=0)
        b_pred = np.expand_dims(y_pred, axis=0)
        chamfer = self.evaluate(ops.multi_chamfer(b_true, b_pred))
        np.testing.assert_allclose(chamfer, [expected_chamfer], atol=1e-5)


if __name__ == '__main__':
    tf.test.main()
    # OpsTest().test_multi_chamfer()
