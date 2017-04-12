#!/usr/bin/env python

import argparse
import os
import pickle
import sys
import numpy as np
import tensorflow as tf
from projlearn import *

parser = argparse.ArgumentParser(description='Prediction.')
parser.add_argument('--kmeans', default='kmeans.pickle', nargs='?', help='Path to k-means.pickle.')
parser.add_argument('--model',  default='baseline', nargs='?', choices=MODELS.keys(), help='The model.')
parser.add_argument('--path',   default='', nargs='?', help='The path to the model dump.')
args = parser.parse_args()

kmeans = pickle.load(open(args.kmeans, 'rb'))
print('The number of clusters is %d.' % kmeans.n_clusters, flush=True, file=sys.stderr)

vectors = np.loadtxt(sys.stdin)

X_all, Y_all = vectors[:, :vectors.shape[1]//2], vectors[:, vectors.shape[1]//2:]

assert X_all.shape == Y_all.shape

size = X_all.shape[1]

print('The vector size is %d.' % size, flush=True, file=sys.stderr)

offsets = Y_all - X_all

X_clusters_list = list(enumerate(kmeans.predict(offsets)))

X_clusters = {}

for cluster in range(kmeans.n_clusters):
    X_clusters[cluster] = [i for i, c in X_clusters_list if c == cluster]

model = MODELS[args.model](x_size=size, y_size=size, w_stddev=0, lambda_=0)

Y_hat_all = np.empty(X_all.shape)

for cluster, indices in X_clusters.items():
    with tf.Session() as sess:
        saver = tf.train.Saver()

        saver.restore(sess, os.path.join(args.path, '%s.k%d.trained') % (args.model, cluster + 1))

        Y_hat = sess.run(model.Y_hat, feed_dict={model.X: X_all[indices]})

        for i, j in enumerate(indices):
            Y_hat_all[j] = Y_hat[i]

np.savetxt(sys.stdout.buffer, Y_hat_all)
