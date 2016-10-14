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

def compute_ats(measures):
    return [sum(measures[j].values()) / len(subsumptions_test) for j in range(len(measures))]

def compute_auc(ats):
    return sum([ats[j] + ats[j + 1] for j in range(0, len(ats) - 1)]) / 2 / 10

measures = [{} for _ in range(0, 10)]

for i, (hyponym, hypernym) in enumerate(subsumptions_test):
    actual  = [w for w, _ in w2v.most_similar(positive=[w2v[hyponym]], topn=10)]

    for j in range(0, len(measures)):
        measures[j][(hyponym, hypernym)] = 1. if hypernym in actual[:j + 1] else 0.

    if (i + 1) % 100 == 0:
        ats = compute_ats(measures)
        auc = compute_auc(ats)
        ats_string = ', '.join(['A@%d=%.6f' % (j + 1, ats[j]) for j in range(len(ats))])
        print('%d examples out of %d done for identity: %s. AUC=%.6f.' % (
            i + 1,
            len(subsumptions_test),
            ats_string,
            auc),
        file=sys.stderr, flush=True)

ats = [sum(measures[j].values()) / len(subsumptions_test) for j in range(len(measures))]
auc = sum([ats[j] + ats[j + 1] for j in range(0, len(ats) - 1)]) / 2 / 10
ats_string = ', '.join(['A@%d=%.4f' % (j + 1, ats[j]) for j in range(len(ats))])
print('For identity: overall %s. AUC=%.6f.' % (ats_string, auc), flush=True)
