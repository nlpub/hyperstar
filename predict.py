#!/usr/bin/env python

import argparse
import os
import pickle
from gensim.models.word2vec import Word2Vec
import csv
import sys
import numpy as np
import tensorflow as tf
from projlearn import *

parser = argparse.ArgumentParser(description='Prediction.')
parser.add_argument('--w2v',    required=True, type=argparse.FileType('rb'))
parser.add_argument('--kmeans', default='kmeans.pickle', nargs='?', help='Path to k-means.pickle.')
parser.add_argument('--model',  default='baseline', nargs='?', choices=MODELS.keys(), help='The model.')
parser.add_argument('--path',   default='', nargs='?', help='The path to the model dump.')
parser.add_argument('output',   type=argparse.FileType('wb'), help='The output npz file.')
args = parser.parse_args()

w2v = Word2Vec.load_word2vec_format(args.w2v, binary=True, unicode_errors='ignore')
w2v.init_sims(replace=True)

print('Using %d word2vec dimensions from "%s".' % (w2v.layer1_size, args.w2v.name), flush=True, file=sys.stderr)

kmeans = pickle.load(open(args.kmeans, 'rb'))

print('The number of clusters is %d.' % kmeans.n_clusters, flush=True, file=sys.stderr)

reader = csv.reader(sys.stdin, delimiter='\t', quoting=csv.QUOTE_NONE)

X_all, Y_all = [], []

for row in reader:
    X_all.append(w2v[row[0]])
    Y_all.append(w2v[row[1]])

X_all, Y_all = np.array(X_all), np.array(Y_all)

offsets = Y_all - X_all

X_clusters_list = list(enumerate(kmeans.predict(offsets)))

X_clusters = {}

for cluster in range(kmeans.n_clusters):
    X_clusters[cluster] = [i for i, c in X_clusters_list if c == cluster]

model = MODELS[args.model](x_size=w2v.layer1_size, y_size=w2v.layer1_size, w_stddev=0, lambda_=0)

Y_hat_all = np.empty(X_all.shape)

for cluster, indices in X_clusters.items():
    with tf.Session() as sess:
        saver = tf.train.Saver()

        saver.restore(sess, os.path.join(args.path, '%s.k%d.trained') % (args.model, cluster + 1))

        Y_hat = sess.run(model.Y_hat, feed_dict={model.X: X_all[indices]})

        for i, j in enumerate(indices):
            Y_hat_all[j] = Y_hat[i]

np.savez_compressed(args.output, Y_hat_all=Y_hat_all)
