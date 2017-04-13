#!/usr/bin/env python

import argparse
import os
import pickle
from gensim.models.word2vec import Word2Vec
import sys
import csv
from itertools import zip_longest
import numpy as np
import tensorflow as tf
from projlearn import *
from gzip import GzipFile

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

parser = argparse.ArgumentParser(description='Prediction.')
parser.add_argument('--w2v',    required=True, type=argparse.FileType('rb'))
parser.add_argument('--kmeans', default='kmeans.pickle', nargs='?', help='Path to k-means.pickle.')
parser.add_argument('--model',  default='baseline', nargs='?', choices=MODELS.keys(), help='The model.')
parser.add_argument('--path',   default='', nargs='?', help='The path to the model dump.')
parser.add_argument('--slices', default=100000, type=int, help='The slice size.')
parser.add_argument('--gzip', default=False, action='store_true')
parser.add_argument('output',   type=argparse.FileType('wb'), help='Output file.')
args = parser.parse_args()

w2v = Word2Vec.load_word2vec_format(args.w2v, binary=True, unicode_errors='ignore')
w2v.init_sims(replace=True)

print('Using %d word2vec dimensions from "%s".' % (w2v.layer1_size, args.w2v.name), flush=True, file=sys.stderr)

kmeans = pickle.load(open(args.kmeans, 'rb'))

print('The number of clusters is %d.' % kmeans.n_clusters, flush=True, file=sys.stderr)

model = MODELS[args.model](x_size=w2v.layer1_size, y_size=w2v.layer1_size, w_stddev=0, lambda_=0)

reader = csv.reader(sys.stdin, delimiter='\t', quoting=csv.QUOTE_NONE)

with args.output if not args.gzip else GzipFile(fileobj=args.output, mode='wb') as f:
    for s, rows in enumerate(grouper(args.slices, reader)):
        X_all, Y_all = [], []

        for row in rows:
            if row is None:
                continue

            X_all.append(w2v.wv.vocab[row[0]].index)
            Y_all.append(w2v.wv.vocab[row[1]].index)

        X_all, Y_all = w2v.wv.syn0[X_all], w2v.wv.syn0[Y_all]

        offsets = Y_all - X_all

        X_clusters_list = list(enumerate(kmeans.predict(offsets)))

        X_clusters = {}

        for cluster in range(kmeans.n_clusters):
            X_clusters[cluster] = [i for i, c in X_clusters_list if c == cluster]

        Y_hat_all = np.empty(X_all.shape)

        for cluster, indices in X_clusters.items():
            with tf.Session() as sess:
                saver = tf.train.Saver()

                saver.restore(sess, os.path.join(args.path, '%s.k%d.trained') % (args.model, cluster + 1))

                Y_hat = sess.run(model.Y_hat, feed_dict={model.X: X_all[indices]})

                for i, j in enumerate(indices):
                    Y_hat_all[j] = Y_hat[i]

        np.save(f, Y_hat_all, allow_pickle=False)

        print('%d slices done.' % (s + 1), flush=True, file=sys.stderr)
