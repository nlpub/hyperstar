import tensorflow as tf
from baseline import Baseline, BaselineCosine

class PositiveHypernym(Baseline):
    """
    A positive sampling approach that promotes the projection matrix to
    project not just the initial hyponym, but its synonyms as well.
    """
    def __init__(self, x_size, y_size, alpha=.3):
        super().__init__(x_size, y_size)

        self.alpha    = alpha

        self.ZY_hat   = tf.matmul(self.Z, self.W)

        self.ZY_error = tf.sub(self.ZY_hat, self.Y)
        self.ZY_loss  = tf.nn.l2_loss(self.ZY_error)

        self.loss     = tf.add((1 - self.alpha) * self.Y_loss, self.alpha * self.ZY_loss)

class PositiveHypernymCosine(BaselineCosine):
    """
    A positive sampling approach that promotes the projection matrix to
    project not just the initial hyponym, but its synonyms as well.
    Cosine is used as the loss function.
    """
    def __init__(self, x_size, y_size, alpha=.3):
        super().__init__(x_size, y_size)

        self.alpha             = alpha

        self.ZY_hat            = tf.matmul(self.Z, self.W)
        self.ZY_hat_normalized = tf.nn.l2_normalize(self.ZY_hat, dim=1)

        self.ZY_cosine         = tf.diag_part(tf.matmul(self.Y_normalized, self.ZY_hat_normalized, transpose_b=True))
        self.ZY_loss           = tf.nn.l2_loss(tf.add(self.ZY_cosine, -1))

        self.loss              = tf.add((1 - self.alpha) * self.Y_loss, self.alpha * self.ZY_loss)
