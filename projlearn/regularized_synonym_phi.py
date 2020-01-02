__author__ = 'Dmitry Ustalov'

import tensorflow as tf

from .regularized_hyponym_phi import RegularizedHyponymPhi


class RegularizedSynonymPhi(RegularizedHyponymPhi):
    """
    A regularized loss function that penalizes the symmetry of
    the projection matrix using the synonyms of the initial hyponym.
    """

    def __init__(self, x_size, y_size, w_stddev, **kwargs):
        super().__init__(x_size, y_size, w_stddev, **kwargs)

        self.YY_similarity = self.dot(self.Y_hat, self.Z)
        self.YY_loss = tf.nn.l2_loss(self.YY_similarity) * 2

        self.loss = tf.add(self.Y_loss, self.lambda_ * self.YY_loss)
