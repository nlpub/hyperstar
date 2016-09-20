#!/usr/bin/env python

import sys, csv, random
from gensim.models.word2vec import Word2Vec
import numpy as np

RANDOM_SEED = 228
random.seed(RANDOM_SEED)

w2v = Word2Vec.load_word2vec_format('all.norm-sz100-w10-cb0-it1-min100.w2v', binary=True, unicode_errors='ignore')
w2v.init_sims(replace=True)

with np.load('test-vectors.npz') as data:
    X_all_test, Y_all_test = data['X_all_test'], data['Y_all_test']

subsumptions_test = []

with open('subsumptions-test.txt') as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        subsumptions_test.append((row[0], row[1]))

accuracies = {}

for i, (hyponym, hypernym) in enumerate(subsumptions_test):
    actual  = {w for w, _ in w2v.most_similar(positive=[w2v[hyponym]], topn=10)}
    measure = 1. if hypernym in actual else 0.
    accuracies[(hyponym, hypernym)] = measure

    if (i + 1) % 100 == 0:
        print('%d examples out of %d done for the identity setting.' % (i + 1, len(subsumptions_test)), file=sys.stderr)

print('Overall A@10 is %.4f %%.' % (sum(accuracies.values()) / len(accuracies) * 100))
