#!/usr/bin/env python

from batch_sim.nn_vec import nn_vec
import argparse
import csv
import glob
import os
import pickle
import re
import sys
from gensim.models.word2vec import Word2Vec
from collections import defaultdict
import numpy as np
import tensorflow as tf
from projlearn import *

parser = argparse.ArgumentParser(description='Expansion.')
parser.add_argument('--kmeans',     default='kmeans.pickle', nargs='?', help='Path to k-means.pickle.')
parser.add_argument('--model',      default='baseline', nargs='?', choices=MODELS.keys(), help='The model.')
parser.add_argument('--path',       default='', nargs='?', help='The path to the model dump.')
parser.add_argument('--w2v',        default='all.norm-sz100-w10-cb0-it1-min100.w2v', nargs='?', help='Path to the word2vec model.')
parser.add_argument('--slow',       action='store_true', help='Disable most similar words calculation optimization.')
parser.add_argument('--neighbours', nargs='?', type=int, default=10)
parser.add_argument('subsumptions', help='Subsumption pairs to expand.')
args = vars(parser.parse_args())

kmeans = pickle.load(open(args['kmeans'], 'rb'))
print('The number of clusters is %d.' % (kmeans.n_clusters), flush=True, file=sys.stderr)

w2v = Word2Vec.load_word2vec_format(args['w2v'], binary=True, unicode_errors='ignore')
w2v.init_sims(replace=True)
print('Using %d word2vec dimensions from "%s".' % (w2v.layer1_size, args['w2v']), flush=True, file=sys.stderr)

subsumptions = []

with open(args['subsumptions']) as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)

    for row in reader:
        if row[0] in w2v and row[1] in w2v:
            subsumptions.append((row[0], row[1]))
        else:
            print('Pair (%s, %s) is not present in w2v.' % row, flush=True, file=sys.stderr)

X_all = np.array([w2v[word] for word, _ in subsumptions])
Y_all = np.array([w2v[word] for _, word in subsumptions])

offsets = Y_all - X_all

X_clusters_list = list(enumerate(kmeans.predict(offsets)))

X_clusters = {}

for cluster in range(kmeans.n_clusters):
    X_clusters[cluster] = [i for i, c in X_clusters_list if c == cluster]

model = MODELS[args['model']](x_size=w2v.layer1_size, y_size=w2v.layer1_size, w_stddev=0, lambda_=0)

Y_hat_all = [None] * X_all.shape[0]

for cluster, indices in X_clusters.items():
    with tf.Session() as sess:
        saver = tf.train.Saver()

        saver.restore(sess, os.path.join(args['path'], '%s.k%d.trained') % (args['model'], cluster + 1))

        Y_hat = sess.run(model.Y_hat, feed_dict={model.X: X_all[indices]})

        for i, j in enumerate(indices):
            Y_hat_all[j] = Y_hat[i]

expansions = {}

if not args['slow']:
    Y_hat_all_norm  = Y_hat_all / np.linalg.norm(Y_hat_all, axis=1)[:, np.newaxis]
    indices, similarities = nn_vec(Y_hat_all_norm, w2v.syn0norm, topn=args['neighbours'], sort=True, return_sims=True, nthreads=8, verbose=False)
    similar_words = [[(w2v.index2word[index], similarities[i][j]) for j, index in enumerate(neighbours)] for i, neighbours in enumerate(indices)]

for i, pair in enumerate(subsumptions):
    if not args['slow']:
        expansions[pair] = similar_words[i]
    else:
        expansions[pair] = w2v.most_similar(positive=[Y_hat_all[i]], topn=args['neighbours'])

for (hyponym, hypernym), expansion in expansions.items():
    print('\t'.join((hyponym, hypernym, ', '.join(('%s:%f' % entry for entry in expansion)))))
