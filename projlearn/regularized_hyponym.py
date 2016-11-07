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

        self.YY_hat        = tf.matmul(self.Y_hat, self.W)

        self.YY_similarity = tf.diag_part(tf.matmul(self.X, self.YY_hat, transpose_b=True))
        self.YY_loss       = tf.nn.l2_loss(self.YY_similarity)

        self.loss          = tf.add(self.Y_loss, self.lambda_ * self.YY_loss)

    def __str__(self):
        return '<%s lambda=%f>' % (self.__class__.__name__, self.lambda_)
