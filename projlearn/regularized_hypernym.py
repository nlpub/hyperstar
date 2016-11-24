import tensorflow as tf
from .baseline import Baseline

class RegularizedHypernym(Baseline):
    """
    A regularized loss function that promotes the projection matrix to
    project not just the initial hyponym, but its synonyms as well.
    """
    def __init__(self, x_size, y_size, w_stddev, **kwargs):
        super().__init__(x_size, y_size, w_stddev, **kwargs)

        self.lambda_  = kwargs['lambda_']

        self.ZY_hat   = tf.matmul(self.Z, self.W)

        self.ZY_error = tf.sub(self.ZY_hat, self.Y)
        self.ZY_loss  = tf.nn.l2_loss(self.ZY_error)

        self.loss     = tf.add(self.Y_loss, self.lambda_ * self.ZY_loss)
