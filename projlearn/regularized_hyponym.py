import tensorflow as tf
from .baseline import Baseline

class RegularizedHyponym(Baseline):
    '''
    A regularized loss function that penalizes the symmetry of
    the projection matrix using the initial hyponym.
    '''
    def __init__(self, x_size, y_size, w_stddev, **kwargs):
        super().__init__(x_size, y_size, w_stddev, **kwargs)

        self.lambda_       = kwargs['lambda_']

        self.YY_hat        = tf.matmul(self.Y_hat, self.W)

        self.YY_similarity = self.dot(self.X, self.YY_hat)
        self.YY_loss       = tf.nn.l2_loss(self.YY_similarity)

        self.loss          = tf.add(self.Y_loss, self.lambda_ * self.YY_loss)

    def __str__(self):
        return '<%s lambda=%f>' % (self.__class__.__name__, self.lambda_)

    def dot(self, X, Y, name='dot_op'):
        with tf.name_scope(name) as scope:
            dot_op = tf.diag_part(tf.matmul(X, Y, transpose_b=True))
            return dot_op
