import tensorflow as tf
from .regularized_hyponym import RegularizedHyponym

class RegularizedSynonym(RegularizedHyponym):
    '''
    A regularized loss function that penalizes the symmetry of
    the projection matrix using the synonyms of the initial hyponym.
    '''
    def __init__(self, x_size, y_size, **kwargs):
        super().__init__(x_size, y_size, **kwargs)

        self.YY_similarity = tf.diag_part(tf.matmul(self.Z[:, 1:], self.YY_hat, transpose_b=True))
        self.YY_loss       = tf.nn.l2_loss(self.YY_similarity)

        self.loss          = tf.add(self.Y_loss, self.lambda_ * self.YY_loss)
