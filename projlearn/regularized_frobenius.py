import tensorflow as tf
from .baseline import Baseline

class RegularizedFrobenius(Baseline):
    """
    Using the Frobenius norm for regularizing the projection matrix of
    the baseline approach.
    """
    def __init__(self, x_size, y_size, **kwargs):
        super().__init__(x_size, y_size, **kwargs)

        self.lambda_  = kwargs['lambda_']

        self.F_norm = tf.sqrt(tf.trace(tf.matmul(self.W, tf.transpose(self.W))))

        self.loss   = tf.add(self.Y_loss, self.lambda_ * self.F_norm) / tf.to_float(tf.shape(self.X)[0])
