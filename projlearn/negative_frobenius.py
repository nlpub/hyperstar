import tensorflow as tf
from baseline import Baseline, BaselineCosine

class NegativeFrobenius(Baseline):
    """
    Using the Frobenius norm for regularizing the projection matrix of
    the baseline approach.
    """
    def __init__(self, x_size, y_size, alpha=.3):
        super().__init__(x_size, y_size)

        self.alpha  = alpha

        self.F_norm = tf.sqrt(tf.trace(tf.matmul(self.W, tf.transpose(self.W))))

        self.loss   = tf.add((1 - self.alpha) * self.Y_loss, self.alpha * self.F_norm)

class NegativeFrobeniusCosine(BaselineCosine):
    """
    Using the Frobenius norm for regularizing the projection matrix of
    the baseline approach. Cosine is used as the loss function.
    """
    def __init__(self, x_size, y_size, alpha=.15):
        super().__init__(x_size, y_size)

        self.alpha  = alpha

        self.F_norm = tf.sqrt(tf.trace(tf.matmul(self.W, tf.transpose(self.W))))

        self.loss   = tf.add((1 - self.alpha) * self.Y_loss, self.alpha * self.F_norm)
