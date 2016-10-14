import tensorflow as tf
from .baseline import Baseline

class RegularizedHyponym(Baseline):
    '''
    A regularized loss function that penalizes the symmetry of
    the projection matrix using the initial hyponym.
    '''
    def __init__(self, x_size, y_size, **kwargs):
        super().__init__(x_size, y_size, **kwargs)

        self.lambda_       = kwargs['lambda_']

        self.YX_hat        = tf.concat(1, [tf.ones((tf.shape(self.Y_hat)[0], 1)), self.Y_hat])
        self.YY_hat        = tf.matmul(self.YX_hat, self.W)

        self.YY_similarity = tf.diag_part(tf.matmul(self.X[:, 1:], self.YY_hat, transpose_b=True))
        self.YY_loss       = tf.nn.l2_loss(self.YY_similarity)

        self.loss          = tf.add(self.Y_loss, self.lambda_ * self.YY_loss)
