import tensorflow as tf
from conditional import conditional

class Toyota:
    """
    A simple baseline that estimates the projection matrix W
    given the vectors X and Y without any constraints.
    """
    def __init__(self, embs_type, embs_shape, cpuembs, w_stddev, **kwargs):
        embs_dim = embs_shape[1]
        with conditional(cpuembs, tf.device('/cpu:0')):
            self.init_embs_subgraph(embs_type, embs_shape)
       
        # Inputs and vars 
        with conditional(cpuembs, tf.device('/cpu:0')):
            x_vec_ph = tf.placeholder(embs_type, [None, embs_dim], name='x_vec_ph')
            y_ind_ph = tf.placeholder(tf.int32, [None, 1], name='y_ind_ph')
            x = x_vec_ph  # n x dim #TODO: add x_ind placeholder and use EmbeddingsLookup(y_ind) instead of placeholder?
            y_ind = y_ind_ph[:,0]  # n
            y = tf.nn.embedding_lookup(self.embs_var, y_ind, name='y_embs_lookup')
#            print(self.embs_var.get_shape(), self.embs_var.dtype)
            print(y.get_shape(), y.dtype)

        # Predict hypernym vector from hyponym vector
        with tf.name_scope('xW'):
            W = tf.Variable(tf.random_normal([embs_dim, embs_dim], stddev=w_stddev), name='W')  # dim x dim
            y_hat = tf.matmul(x, W)  # n x dim

        # Dot similarity of predicted hypernym with true hypernym
        with tf.name_scope('score_yhat_y'):
            pos_logit = tf.reduce_sum(y_hat * y, axis=1)  # n
            print(pos_logit.get_shape(), pos_logit.dtype)

        # Dot similarity of predicted hypernym with all words
        with tf.name_scope('score_yhat_embs'), conditional(cpuembs, tf.device('/cpu:0')):
            dot = tf.matmul(y_hat, self.embs_var, transpose_b=True)  # n x V

        with tf.name_scope('negative_ind'):
            top_k_vals, top_k_inds = tf.nn.top_k(dot, k=2, sorted=True)  # n x 2
            print(top_k_vals.get_shape())
            # takes all top_k y' which are not equal to true y
            # TODO: try also taking only first y' not equal to true y
            top_mask = tf.not_equal(y_ind, top_k_inds[:, 0], name='top_mask')  # n
            print(top_mask.get_shape())
            neg_ind = tf.where(top_mask, top_k_inds[:, 0], top_k_inds[:, 1])  # n
#            neg_logit = tf.where(top_mask, top_k_vals[:, 0], top_k_vals[:, 1])  # n
#            print(neg_logit.get_shape())

        with tf.name_scope('score_yhat_yneg'):
            with conditional(cpuembs, tf.device('/cpu:0')):
                y_neg = tf.nn.embedding_lookup(self.embs_var, neg_ind, name='y_neg_embs_lookup')
            neg_logit = tf.reduce_sum(y_hat * tf.stop_gradient(y_neg), axis=1) # n
            print(neg_logit.get_shape())

        # Loss
        with tf.name_scope('loss'):
            with tf.name_scope('pos_loss'):
                pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(pos_logit,
                                                                   tf.constant(1, dtype=tf.float32, shape=[1]))
            with tf.name_scope('neg_loss'):
                neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(neg_logit,
                                                                   tf.constant(0, dtype=tf.float32, shape=[1]))
            loss = tf.reduce_mean(pos_loss + neg_loss)
        print(pos_loss.get_shape(), neg_loss.get_shape(), loss.get_shape())

        # Evaluation
        with tf.name_scope('evaluation'):
            tf.summary.histogram('W', W)
            tf.summary.histogram('neg_logit', neg_logit)
            tf.summary.histogram('pos_logit', pos_logit)
            tf.summary.scalar('avg_neg_logit', tf.reduce_mean(neg_logit))
            tf.summary.scalar('avg_pos_logit', tf.reduce_mean(pos_logit))
            tf.summary.scalar('avg_neg_LOSS', tf.reduce_mean(neg_loss))
            tf.summary.scalar('avg_pos_LOSS', tf.reduce_mean(pos_loss))
            tf.summary.scalar('LOSS', loss)
            acc_10 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(dot, y_ind, k=10), tf.float32))
            acc_2 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(dot, y_ind, k=2), tf.float32))
            tf.summary.scalar('acc_10', acc_10)
            tf.summary.scalar('acc_2', acc_2)

        # Summaries
        self.summary = tf.summary.merge_all()
        self.X = x_vec_ph
        self.Y = y_ind_ph
        self.Z = tf.placeholder(tf.float32, shape=[None, embs_dim], name='Z') # not used, for compatibility

        self.Y_hat = y_hat
        self.loss = loss


    def init_embs_subgraph(self, embs_type, embs_shape):
        # move embs to Variable
        with tf.name_scope('init_embs_var'):
            embs_ph = tf.placeholder(embs_type, embs_shape, name='embs_ph')
            embs_var = tf.get_variable('embs_var', shape=embs_shape, dtype=embs_type, trainable=False)
            embs_assign = tf.assign(embs_var, embs_ph, name='assign_embs_var')
            self.embs_ph, self.embs_var, self.embs_assign = embs_ph, embs_var, embs_assign

 
    def __str__(self):
        return '<%s>' % self.__class__.__name__

    def load_w2v(self, embs, sess):
        sess.run(self.embs_assign, feed_dict={self.embs_ph: embs})


    def init_summary(self):
        pass # self.summary assigned in __init__
