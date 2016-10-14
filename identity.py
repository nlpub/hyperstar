#!/usr/bin/env python

import csv
import random
import sys
from gensim.models.word2vec import Word2Vec
import numpy as np

RANDOM_SEED = 228
random.seed(RANDOM_SEED)

w2v = Word2Vec.load_word2vec_format('all.norm-sz100-w10-cb0-it1-min100.w2v', binary=True, unicode_errors='ignore')
w2v.init_sims(replace=True)

with np.load('test.npz') as npz:
    Y_all_test    = npz['Y_all_test']
    Z_index_test  = npz['Z_index_test']
    Z_all_test    = npz['Z_all_test']

X_all_test  = Z_all_test[Z_index_test[:, 0], :]

subsumptions_test = []

with open('subsumptions-test.txt') as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        subsumptions_test.append((row[0], row[1]))

measures = [{} for _ in range(0, 10)]

for i, (hyponym, hypernym) in enumerate(subsumptions_test):
    actual  = [w for w, _ in w2v.most_similar(positive=[w2v[hyponym]], topn=10)]

    for j in range(0, len(measures)):
        measures[j][(hyponym, hypernym)] = 1. if hypernym in actual[:j + 1] else 0.

    if (i + 1) % 100 == 0:
        print('%d examples out of %d done for identity: %s.' % (i + 1,
            len(subsumptions_test),
            ', '.join(['@%d=%.6f' % (i + 1, sum(measures[i].values()) / len(subsumptions_test)) for i in range(len(measures))])),
            file=sys.stderr, flush=True)

print('For identity: overall %s.' % (', '.join(['@%d=%.4f' % (i + 1, sum(measures[i].values()) / len(subsumptions_test)) for i in range(len(measures))])), flush=True)
