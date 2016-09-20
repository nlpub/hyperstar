import tensorflow as tf
from baseline import Baseline, BaselineCosine

class NegativeHypernym(Baseline):
    """
    A negative sampling approach, based on the baseline, which
    tries to penalize the symmetry of the projection matrix.
    """
    def __init__(self, x_size, y_size, alpha=.01):
        super().__init__(x_size, y_size)

        self.alpha   = alpha
        self.X_hat   = tf.matmul(self.Y_hat, self.W, transpose_b=True)
        self.X_error = tf.sub(self.X_hat, self.X)
        self.X_loss  = tf.nn.l2_loss(self.X_error)
        self.loss    = tf.abs(tf.sub((1 - self.alpha) * self.Y_loss, self.alpha * self.X_loss))

class NegativeHypernymCosine(BaselineCosine):
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
        self.X_normalized     = tf.nn.l2_normalize(self.X,     dim=1)
        self.X_cosine         = tf.diag_part(tf.matmul(self.X_normalized, self.X_hat_normalized, transpose_b=True))
        self.X_loss           = tf.nn.l2_loss(tf.add(self.X_cosine, +1))
        self.loss             = tf.add((1 - self.alpha) * self.Y_loss, self.alpha * self.X_loss)
