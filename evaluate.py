#!/usr/bin/env python

import csv
import glob
import os
import pickle
import random
import re
import sys
from gensim.models.word2vec import Word2Vec
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans

RANDOM_SEED = 228
random.seed(RANDOM_SEED)

MODELS = ['baseline', 'negative_hyponym', 'negative_synonym', 'positive_hypernym']

if not len(sys.argv) > 1:
    print('Usage: %s path...' % (sys.argv[0]), file=sys.stderr)
    sys.exit(1)

WD = os.path.dirname(os.path.realpath(__file__))

w2v = Word2Vec.load_word2vec_format(os.path.join(WD, 'all.norm-sz100-w10-cb0-it1-min100.w2v'), binary=True, unicode_errors='ignore')
w2v.init_sims(replace=True)

with np.load('train.npz') as npz:
    Y_all_train   = npz['Y_all_train']
    Z_index_train = npz['Z_index_train']
    Z_all_train   = npz['Z_all_train']

with np.load('test.npz') as npz:
    Y_all_test    = npz['Y_all_test']
    Z_index_test  = npz['Z_index_test']
    Z_all_test    = npz['Z_all_test']

X_all_train = Z_all_train[Z_index_train[:, 0], :]
X_all_test  = Z_all_test[Z_index_test[:, 0],   :]

subsumptions_test = []

with open('subsumptions-test.txt') as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        subsumptions_test.append((row[0], row[1]))

for path in sys.argv[1:]:
    print('Doing "%s".' % path, flush=True)

    kmeans = pickle.load(open(os.path.join(path, 'kmeans.pickle'), 'rb'))
    print('The number of clusters is %d.' % (kmeans.n_clusters), flush=True)

    for model in MODELS:
        clusters_train = kmeans.predict(Y_all_train - X_all_train)
        clusters_test  = kmeans.predict(Y_all_test  - X_all_test)

        W = [None] * kmeans.n_clusters

        CLUSTER_REGEXP = re.compile('W-(?P<cluster>\d+)\.txt$')

        for model_path in glob.glob('%s/%s.W-*.txt' % (path, model)):
            cluster = int(CLUSTER_REGEXP.search(model_path).group('cluster')) - 1
            print('Loading "%s" as the cluster %d.' % (model_path, cluster), flush=True)
            W[cluster] = np.loadtxt(model_path)

        measures = [{} for _ in range(0, 10)]
        cache = defaultdict(lambda: {})

        for i, (hyponym, hypernym) in enumerate(subsumptions_test):
            cluster   = clusters_test[i]

            if hyponym not in cache[cluster]:
                X_example = np.ones((1, X_all_test.shape[1] + 1))
                X_example[:, 1:] = w2v[hyponym]
                Y_example = X_example.dot(W[cluster]).reshape(X_all_test.shape[1],)
                cache[cluster][hyponym] = [w for w, _ in w2v.most_similar(positive=[Y_example], topn=10)]

            actual  = cache[cluster][hyponym]

            for j in range(0, len(measures)):
                measures[j][(hyponym, hypernym)] = 1. if hypernym in actual[:j + 1] else 0.

            if (i + 1) % 100 == 0:
                print('%d examples out of %d done for "%s/%s": %s.' % (i + 1,
                    len(subsumptions_test), path, model,
                    ', '.join(['A@%d=%.6f' % (i + 1, sum(measures[i].values()) / len(subsumptions_test)) for i in range(len(measures))])),
                    file=sys.stderr, flush=True)

        print('For "%s/%s": overall %s.' % (
            path, model,
            ', '.join(['A@%d=%.4f' % (i + 1, sum(measures[i].values()) / len(subsumptions_test)) for i in range(len(measures))])),
            flush=True)
