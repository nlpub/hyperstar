#!/usr/bin/env python

import os, glob, sys, csv, math, re, random, pickle
from gensim.models.word2vec import Word2Vec
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from multiprocessing import Pool

RANDOM_SEED = 228
random.seed(RANDOM_SEED)

if not len(sys.argv) > 1:
    print('Usage: %s model' % (sys.argv[0]), file=sys.stderr)
    sys.exit(1)

MODEL = sys.argv[1]

w2v = Word2Vec.load_word2vec_format('all.norm-sz100-w10-cb0-it1-min100.w2v', binary=True, unicode_errors='ignore')
w2v.init_sims(replace=True)
e_size = w2v.layer1_size

with np.load('train-vectors.npz') as data:
    X_all_train, Y_all_train = data['X_all_train'], data['Y_all_train']

with np.load('test-vectors.npz') as data:
    X_all_test, Y_all_test = data['X_all_test'], data['Y_all_test']

subsumptions_test = []

with open('subsumptions-test.txt') as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        subsumptions_test.append((row[0], row[1]))

kmeans = pickle.load(open('kmeans.pickle', 'rb'))
print('The number of clusters is %d.' % (kmeans.n_clusters))

clusters_train = kmeans.predict(Y_all_train - X_all_train)
clusters_test  = kmeans.predict(Y_all_test  - X_all_test)

W = [None] * kmeans.n_clusters

CLUSTER_REGEXP = re.compile('W-(?P<cluster>\d+)\.txt$')

for path in glob.glob('%s.W-*.txt' % (MODEL)):
    cluster = int(CLUSTER_REGEXP.search(path).group('cluster')) - 1
    print('Loading "%s" as the cluster %d.' % (path, cluster))
    W[cluster] = np.loadtxt(path)

measures1, measures5, measures10 = {}, {}, {}
cache = defaultdict(lambda: {})

for i, (hyponym, hypernym) in enumerate(subsumptions_test):
    cluster   = clusters_test[i]

    if hyponym not in cache[cluster]:
        X_example = np.ones((1, X_all_test.shape[1] + 1))
        X_example[:, 1:] = w2v[hyponym]
        Y_example = X_example.dot(W[cluster]).reshape(X_all_test.shape[1],)
        cache[cluster][hyponym] = [w for w, _ in w2v.most_similar(positive=[Y_example], topn=10)]

    actual  = cache[cluster][hyponym]

    measure1 = 1. if hypernym in actual[:1] else 0.
    measures1[(hyponym, hypernym)] = measure1

    measure5 = 1. if hypernym in actual[:5] else 0.
    measures1[(hyponym, hypernym)] = measure5

    measure10 = 1. if hypernym in actual[:10] else 0.
    measures10[(hyponym, hypernym)] = measure10

    if (i + 1) % 100 == 0:
        print('%d examples out of %d done for "%s": A@1 is %.6f, A@5 is %.6f and A@10 is %.6f.' % (i + 1,
            len(subsumptions_test), MODEL,
            sum(measures1.values())  / len(subsumptions_test),
            sum(measures5.values())  / len(subsumptions_test),
            sum(measures10.values()) / len(subsumptions_test)), file=sys.stderr)

print('Overall A@1 is %.4f, A@5 is %.4f and A@10 is %.4f.' % (
    sum(measures1.values())  / len(subsumptions_test),
    sum(measures5.values())  / len(subsumptions_test),
    sum(measures10.values()) / len(subsumptions_test)))
