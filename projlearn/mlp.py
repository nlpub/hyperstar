import tensorflow as tf

class MLP:
    def __init__(self, x_size, y_size, lambda_, **kwargs):
        self.x_size = x_size
        self.y_size = y_size

        self.X = tf.placeholder(tf.float32, shape=[None, self.x_size], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.y_size], name='Y')
        self.Z = tf.placeholder(tf.float32, shape=[None, self.x_size], name='Z')

        self.lambda_       = lambda_

        self.layers = [self.X]

        for i in (5, 100):
            self.layers.append(tf.contrib.layers.fully_connected(inputs=self.layers[-1], num_outputs=i,
                activation_fn=tf.nn.relu,
                weights_initializer=lambda shape, dtype: tf.random_normal(shape, stddev=.1, dtype=dtype),
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.lambda_),
                biases_initializer=lambda shape, dtype: tf.random_normal(shape, stddev=.1, dtype=dtype),
                biases_regularizer=tf.contrib.layers.l2_regularizer(self.lambda_)))

        self.layers.append(tf.contrib.layers.fully_connected(inputs=self.layers[-1], num_outputs=y_size,
                weights_initializer=lambda shape, dtype: tf.random_normal(shape, stddev=.1, dtype=dtype),
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.lambda_),
                biases_initializer=lambda shape, dtype: tf.random_normal(shape, stddev=.1, dtype=dtype),
                biases_regularizer=tf.contrib.layers.l2_regularizer(self.lambda_)))

        self.Y_hat  = self.layers[-1]

        self.Y_error = tf.sub(self.Y_hat, self.Y)
        self.Y_loss  = tf.nn.l2_loss(self.Y_error)

        self.loss    = self.Y_loss

    def __str__(self):
        return '<%s lambda=%f layers=%d>' % (self.__class__.__name__, self.lambda_, len(self.layers))
