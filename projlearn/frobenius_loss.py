__author__ = 'Dmitry Ustalov'

import tensorflow as tf

from .baseline import Baseline


class FrobeniusLoss(Baseline):
    """
    Using the Frobenius norm as the loss function for the baseline approach.
    """

    def __init__(self, x_size, y_size, w_stddev, **kwargs):
        super().__init__(x_size, y_size, w_stddev, **kwargs)

        self.F_norm = tf.sqrt(tf.trace(tf.matmul(self.Y_error, tf.transpose(self.Y_error))))

        self.loss = self.F_norm
