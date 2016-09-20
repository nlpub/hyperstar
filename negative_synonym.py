import tensorflow as tf
from baseline import Baseline, BaselineCosine

class NegativeSynonym(Baseline):
    """
    A negative sampling approach, based on the baseline, which
    tries to penalize the symmetry of the projection matrix.
    """
    def __init__(self, x_size, y_size, alpha=.01):
        super().__init__(x_size, y_size)

        self.alpha   = alpha
        self.X_hat   = tf.matmul(self.Y_hat, self.W, transpose_b=True)
        self.X_error = tf.sub(self.X_hat, self.Z)
        self.X_loss  = tf.nn.l2_loss(self.X_error)
        self.loss    = tf.abs(tf.sub((1 - self.alpha) * self.Y_loss, self.alpha * self.X_loss))

class NegativeSynonymCosine(BaselineCosine):
    """
    A negative sampling approach, based on the baseline, which
    tries to penalize the symmetry of the projection matrix. Cosine
    is used as the loss function.
    """
    def __init__(self, x_size, y_size, alpha=.01):
        super().__init__(x_size, y_size)

        self.alpha            = alpha
        self.X_hat            = tf.matmul(self.Y_hat, self.W, transpose_b=True)
        self.X_hat_normalized = tf.nn.l2_normalize(self.X_hat, dim=1)
        self.Z_normalized     = tf.nn.l2_normalize(self.Z,     dim=1)
        self.Z_cosine         = tf.diag_part(tf.matmul(self.Z_normalized, self.X_hat_normalized, transpose_b=True))
        self.Z_loss           = tf.nn.l2_loss(tf.add(self.Z_cosine, +1))
        self.loss             = tf.add((1 - self.alpha) * self.Y_loss, self.alpha * self.Z_loss)
