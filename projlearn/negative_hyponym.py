import tensorflow as tf
from .baseline import Baseline, BaselineCosine

class NegativeHyponym(Baseline):
    """
    A negative sampling approach that penalizes the symmetry of
    the projection matrix using the initial hyponym.
    """
    def __init__(self, x_size, y_size, alpha=.01):
        super().__init__(x_size, y_size)

        self.alpha    = alpha

        self.YX_hat   = tf.concat(1, [tf.ones((tf.shape(self.Y_hat)[0], 1)), self.Y_hat])
        self.YY_hat   = tf.matmul(self.YX_hat, self.W)

        self.YY_error = tf.sub(self.YY_hat, self.X[:, 1:])
        self.YY_loss  = tf.nn.l2_loss(self.YY_error)

        self.loss     = tf.abs(tf.sub((1 - self.alpha) * self.Y_loss, self.alpha * self.YY_loss))

class NegativeHyponymCosine(BaselineCosine):
    """
    A negative sampling approach that penalizes the symmetry of
    the projection matrix using the initial hyponym.
    Cosine is used as the loss function.
    """
    def __init__(self, x_size, y_size, alpha=.01):
        super().__init__(x_size, y_size)

        self.alpha             = alpha

        self.YX_hat            = tf.concat(1, [tf.ones((tf.shape(self.Y_hat)[0], 1)), self.Y_hat])
        self.YY_hat            = tf.matmul(self.YX_hat, self.W)
        self.X_normalized      = tf.nn.l2_normalize(self.X[:, 1:], dim=1)
        self.YY_hat_normalized = tf.nn.l2_normalize(self.YY_hat,   dim=1)

        self.YY_cosine         = tf.diag_part(tf.matmul(self.X_normalized, self.YY_hat_normalized, transpose_b=True))
        self.YY_loss           = tf.nn.l2_loss(tf.add(self.YY_cosine, +1))

        self.loss              = tf.add((1 - self.alpha) * self.Y_loss, self.alpha * self.YY_loss)
