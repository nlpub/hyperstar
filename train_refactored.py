#!/usr/bin/env python

import datetime
import glob
import os
import sys
import pickle
import random
import numpy as np
import tensorflow as tf
from projlearn import *
from projlearn.toyota import Toyota
import pandas as pd
import gensim
from tqdm import tqdm 

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string( 'model',  'baseline', 'Model name.')
flags.DEFINE_string( 'train', 'train.npz', 'Training set.')
flags.DEFINE_string( 'test',   'test.npz', 'Test set.')
flags.DEFINE_float(  'stddev',        .01, 'Value of stddev for matrix initialization.')
flags.DEFINE_float(  'lambdac',       .10, 'Value of lambda.')
flags.DEFINE_integer('seed',          228, 'Random seed.')
flags.DEFINE_integer('num_epochs',    300, 'Number of training epochs.')
flags.DEFINE_integer('batch_size',   2048, 'Batch size.')
flags.DEFINE_boolean('gpu',          True, 'Try using GPU.')
flags.DEFINE_boolean('cpuembs',      False, 'Place embedding matrix and ops with it on CPU (instead of soft placement).')
flags.DEFINE_string('w2v',          'corpus_en.norm-sz100-w8-cb0-it1-min20.w2v', 'Path to w2v file (for Toyota model).')
flags.DEFINE_integer('eval_limit', None, 'Maximum number of examples from train/evaluation/test set to evaluate train/test loss etc. during training.')
flags.DEFINE_boolean('log_device_placement',      False, 'Log device placement of nodes in TensorFlow graph.')

MODELS = {
    'baseline':              Baseline,
    'regularized_hyponym':   RegularizedHyponym,
    'regularized_synonym':   RegularizedSynonym,
    'regularized_hypernym':  RegularizedHypernym,
    'frobenius_loss':        FrobeniusLoss,
    'mlp':                   MLP,
    'toyota': Toyota
}

def train(sess, train_op, model, data, callback=lambda: None, train_writer=None, test_writer=None):
    train_losses, test_losses = [], []
    train_times = []

    # Init all vars except embs_var
    init_vars = tf.global_variables()
    if FLAGS.model=='toyota':
        init_vars.remove(model.embs_var)
    sess.run(tf.variables_initializer(init_vars))
    limit = FLAGS.eval_limit
    feed_dict_train, feed_dict_test = {
        model.X: data.X_train[:limit],
        model.Y: data.Y_train[:limit],
        model.Z: data.Z_train[:limit]
    }, {
        model.X: data.X_test[:limit],
        model.Y: data.Y_test[:limit],
        model.Z: data.Z_test[:limit]
    }

    steps = max(data.Y_train.shape[0] // FLAGS.batch_size, 1)

    print('Cluster %d: %d train items and %d test items available; using %d steps of %d items.' % (
        data.cluster + 1,
        data.X_train.shape[0],
        data.X_test.shape[0],
        steps,
        min(FLAGS.batch_size, data.X_train.shape[0])),
    flush=True)

    for epoch in tqdm(range(FLAGS.num_epochs), unit='epoch'):
        X, Y, Z = data.train_shuffle()

        for step in range(steps):
            head =  step      * FLAGS.batch_size
            tail = (step + 1) * FLAGS.batch_size

            feed_dict = {
                model.X: X[head:tail, :],
                model.Y: Y[head:tail, :],
                model.Z: Z[head:tail, :]
            }

            t_this = datetime.datetime.now()
            sess.run(train_op, feed_dict=feed_dict)
            t_last = datetime.datetime.now()

            train_times.append(t_last - t_this)

        if (epoch + 1) % 10 == 0 or (epoch == 0):
            res = sess.run([model.loss, model.summary, model.acc_2, model.acc_10], feed_dict=feed_dict_train)
            train_losses.append(res[0])
            train_writer.add_summary(res[1], epoch)
            res = sess.run([model.loss, model.summary, model.acc_2, model.acc_10], feed_dict=feed_dict_test)
            test_losses.append(res[0])
            test_writer.add_summary(res[1], epoch)

            print('Cluster %d: epoch = %05d, train loss = %f, test loss = %f, test acc_2 = %f, test acc_10 = %f.' % (
                data.cluster + 1,
                epoch + 1,
                train_losses[-1] / data.X_train.shape[0],
                test_losses[-1]  / data.X_test.shape[0],
                res[2], res[3]), 
            file=sys.stderr, flush=True)

    t_delta = sum(train_times, datetime.timedelta())
    print('Cluster %d done in %s.' % (data.cluster + 1, str(t_delta)), flush=True)
    callback(sess)

    return sess.run(model.Y_hat, feed_dict=feed_dict_test)

def main(_):
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    if not FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    config = tf.ConfigProto(log_device_placement = FLAGS.log_device_placement) 

    if FLAGS.model == 'toyota':
        # Load w2v
        embs_type = 'float32'
        # Load word embeddings from W2V file in vectors of tensor_type dtype
        print('Loading w2v model from ', FLAGS.w2v)
        w2v = gensim.models.Word2Vec.load_word2vec_format(FLAGS.w2v, binary=True, unicode_errors='ignore', datatype=embs_type)
        print('Loaded.')
        w2v.init_sims(replace=True)
        embs = w2v.syn0norm

        # Load hypernymy datasets
        dfs = {}

        def load_ds(part):
            f = 'subsumptions-%s.txt' % part
            df = pd.read_csv(f, sep='\t', header=None, names=['hypo', 'hyper'])
            # Convert words to indices
            for col in df.columns:
                df[col + '_ind'] = df[col].apply(lambda x: w2v.vocab[x].index)

            print(f, len(df))
            return df

        dfs['train'] = load_ds('train')
        dfs['test'] = load_ds('validation') if FLAGS.test.endswith('validation.npz') else load_ds('test')


        # get embeddings for hyponym and hypernym
        Y_ind_train = np.array(dfs['train']['hyper_ind'])[:,np.newaxis]
        Y_ind_test = np.array(dfs['test']['hyper_ind'])[:, np.newaxis]



    with np.load(FLAGS.train) as npz:
        X_index_train = npz['X_index']
        Y_all_train   = npz['Y_all']
        Z_all_train   = npz['Z_all']

    with np.load(FLAGS.test) as npz:
        X_index_test  = npz['X_index']
        Y_all_test    = npz['Y_all']
        Z_all_test    = npz['Z_all']

    X_all_train = Z_all_train[X_index_train[:, 0], :]
    X_all_test  = Z_all_test[X_index_test[:, 0],   :]

    kmeans = pickle.load(open('kmeans.pickle', 'rb'))
    clusters_train = kmeans.predict(Y_all_train - X_all_train)
    clusters_test  = kmeans.predict(Y_all_test  - X_all_test)

    if FLAGS.model=='toyota':
        dfs['train']['cluster'] = clusters_train
        dfs['test']['cluster'] = clusters_test

    if FLAGS.model == 'toyota':
        model = Toyota(embs_type, embs.shape, cpuembs=FLAGS.cpuembs,  w_stddev=FLAGS.stddev)
    else:
        model = MODELS[FLAGS.model](x_size=Z_all_train.shape[1], y_size=Y_all_train.shape[1], w_stddev=FLAGS.stddev,
                                    lambda_=FLAGS.lambdac)
        print(model, flush=True)

    model.init_summary()

    for path in glob.glob('%s.k*.trained*' % FLAGS.model):
        print('Removing a stale file: "%s".' % path, flush=True)
        os.remove(path)

    if os.path.isfile('%s.test.npz' % FLAGS.model):
        print('Removing a stale file: "%s".' % ('%s.test.npz' % FLAGS.model), flush=True)
        os.remove('%s.test.npz' % FLAGS.model)

    Y_hat_test = {}

    # Training
    with tf.name_scope('Training'):
        global_step = tf.Variable(tf.constant(0, tf.int32))
        train_op = tf.train.AdamOptimizer().minimize(model.loss, global_step)
    # train_op = tf.train.AdamOptimizer(epsilon=1.).minimize(model.loss)

    with tf.Session(config=config) as sess:
        from datetime import datetime
        t = datetime.now().replace(microsecond=0)

        if FLAGS.model == 'toyota':
            model.load_w2v(embs, sess)

        for cluster in range(kmeans.n_clusters):
            train_writer = tf.summary.FileWriter('./tf_train_logs5/%s-cl%d-train' % (t, cluster), sess.graph)
            test_writer = tf.summary.FileWriter('./tf_train_logs5/%s-cl%d-test' % (t, cluster), sess.graph)

            if FLAGS.model == 'toyota':
                # data = Data_toyota(cluster, dfs['train'], dfs['test'])
                data = Data(
                    cluster, clusters_train, clusters_test,
                    X_index_train, Y_ind_train, Z_all_train,
                    X_index_test, Y_ind_test,  Z_all_test
                )

            else:
                data = Data(
                    cluster, clusters_train, clusters_test,
                    X_index_train, Y_all_train, Z_all_train,
                    X_index_test,  Y_all_test,  Z_all_test
                )

            saver = tf.train.Saver()
            saver_path = '%s.k%d.trained' % (FLAGS.model, cluster + 1)
            Y_hat_test[str(cluster)] = train(sess, train_op, model, data, callback=lambda sess: saver.save(sess, saver_path),
                                             train_writer=train_writer, test_writer=test_writer)
            print('Writing the output model to "%s".' % saver_path, flush=True)

    test_path = '%s.test.npz' % FLAGS.model
    np.savez_compressed(test_path, **Y_hat_test)
    print('Writing the test data to "%s".' % test_path)

if __name__ == '__main__':
    tf.app.run()
