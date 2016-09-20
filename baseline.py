import tensorflow as tf

class Baseline:
    """
    A simple baseline that estimates the projection matrix W
    given the vectors X and Y without any constraints.
    """
    def __init__(self, x_size, y_size):
        self.X = tf.placeholder(tf.float32, shape=[None, x_size + 1], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, y_size], name='Y')
        self.Z = tf.placeholder(tf.float32, shape=[None, x_size + 1], name='Z')
        self.W = tf.Variable(tf.random_normal((x_size + 1, y_size), stddev=0.1), name='W')

        self.Y_hat   = tf.matmul(self.X, self.W)
        self.Y_error = tf.sub(self.Y_hat, self.Y)
        self.Y_loss  = tf.nn.l2_loss(self.Y_error)
        self.loss    = self.Y_loss

class BaselineCosine(Baseline):
    """
    A simple baseline that estimates the projection matrix W
    given the vectors X and Y without any constraints. Cosine
    is used as the loss function.
    """
    def __init__(self, x_size, y_size):
        super().__init__(x_size, y_size)

        self.Y_normalized     = tf.nn.l2_normalize(self.Y,     dim=1)
        self.Y_hat_normalized = tf.nn.l2_normalize(self.Y_hat, dim=1)
        self.Y_cosine         = tf.diag_part(tf.matmul(self.Y_normalized, self.Y_hat_normalized, transpose_b=True))
        self.Y_loss           = tf.nn.l2_loss(tf.add(self.Y_cosine, -1))
        self.loss             = self.Y_loss
