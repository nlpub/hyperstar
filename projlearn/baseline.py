import tensorflow as tf

class Baseline:
    """
    A simple baseline that estimates the projection matrix W
    given the vectors X and Y without any constraints.
    """
    def __init__(self, x_size, y_size, **kwargs):
        self.x_size = x_size
        self.y_size = y_size

        self.X = tf.placeholder(tf.float32, shape=[None, self.x_size], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.y_size], name='Y')
        self.Z = tf.placeholder(tf.float32, shape=[None, self.x_size], name='Z')
        self.W = tf.Variable(tf.random_normal((self.x_size, self.y_size), stddev=0.1), name='W')

        self.Y_hat   = tf.matmul(self.X, self.W)
        self.Y_error = tf.sub(self.Y_hat, self.Y)
        self.Y_loss  = tf.nn.l2_loss(self.Y_error)

        self.loss    = self.Y_loss

    def __str__(self):
        return '<%s>' % self.__class__.__name__
