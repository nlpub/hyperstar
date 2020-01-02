__author__ = 'Dmitry Ustalov'

import tensorflow as tf

from .regularized_hyponym import RegularizedHyponym


class RegularizedSynonym(RegularizedHyponym):
    """
    A regularized loss function that penalizes the symmetry of
    the projection matrix using the synonyms of the initial hyponym.
    Unlike RegularizedSynonymPhi, this approach applies W twice.
    """

    def __init__(self, x_size, y_size, w_stddev, **kwargs):
        super().__init__(x_size, y_size, w_stddev, **kwargs)

        self.YY_similarity = self.dot(self.YY_hat, self.Z)
        self.YY_loss = tf.nn.l2_loss(self.YY_similarity) * 2

        self.loss = tf.add(self.Y_loss, self.lambda_ * self.YY_loss)
