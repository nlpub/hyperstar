#!/usr/bin/env python

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

MODELS = ['baseline', 'regularized_frobenius', 'regularized_hyponym', 'regularized_synonym', 'regularized_hypernym', 'mlp']

parser = argparse.ArgumentParser(description='Evaluation.')
parser.add_argument('--w2v',          default='all.norm-sz100-w10-cb0-it1-min100.w2v', nargs='?', help='Path to the word2vec model.')
parser.add_argument('--test',         default='test.npz',              nargs='?', help='Path to the test set.')
parser.add_argument('--subsumptions', default='subsumptions-test.txt', nargs='?', help='Path to the test subsumptions.')
parser.add_argument('path', nargs='*', help='List of the directories with results.')
args = vars(parser.parse_args())

if not len(sys.argv) > 1:
    print('Usage: %s path...' % (sys.argv[0]), file=sys.stderr)
    sys.exit(1)

WD = os.path.dirname(os.path.realpath(__file__))

w2v = Word2Vec.load_word2vec_format(os.path.join(WD, args['w2v']), binary=True, unicode_errors='ignore')
w2v.init_sims(replace=True)

with np.load(args['test']) as npz:
    Y_all_test    = npz['Y_all']
    Z_index_test  = npz['Z_index']
    Z_all_test    = npz['Z_all']

X_all_test  = Z_all_test[Z_index_test[:, 0],   :]

subsumptions_test = []

with open(args['subsumptions']) as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        subsumptions_test.append((row[0], row[1]))

def compute_ats(measures):
    return [sum(measures[j].values()) / len(subsumptions_test) for j in range(len(measures))]

def compute_auc(ats):
    return sum([ats[j] + ats[j + 1] for j in range(0, len(ats) - 1)]) / 2 / 10

for path in args['path']:
    print('Doing "%s" on "%s" and "%s".' % (path, args['test'], args['subsumptions']), flush=True)

    kmeans = pickle.load(open(os.path.join(path, 'kmeans.pickle'), 'rb'))
    print('The number of clusters is %d.' % (kmeans.n_clusters), flush=True)

    for model in MODELS:
        clusters_test  = kmeans.predict(Y_all_test - X_all_test)

        with np.load('%s.test.npz' % model) as npz:
            Y_hat = {int(cluster): npz[cluster] for cluster in npz.files}

        measures = [{} for _ in range(0, 10)]
        cache = defaultdict(lambda: {})

        for i, (hyponym, hypernym) in enumerate(subsumptions_test):
            cluster   = clusters_test[i]

            if hyponym not in cache[cluster]:
                Y_example = Y_hat[cluster][i].reshape(X_all_test.shape[1],)
                cache[cluster][hyponym] = [w for w, _ in w2v.most_similar(positive=[Y_example], topn=10)]

            actual  = cache[cluster][hyponym]

            for j in range(0, len(measures)):
                measures[j][(hyponym, hypernym)] = 1. if hypernym in actual[:j + 1] else 0.

            if (i + 1) % 100 == 0:
                ats = compute_ats(measures)
                auc = compute_auc(ats)
                ats_string = ', '.join(['A@%d=%.6f' % (j + 1, ats[j]) for j in range(len(ats))])
                print('%d examples out of %d done for "%s/%s": %s. AUC=%.6f.' % (
                    i + 1,
                    len(subsumptions_test),
                    path,
                    model,
                    ats_string,
                    auc),
                file=sys.stderr, flush=True)

        ats = compute_ats(measures)
        auc = compute_auc(ats)
        ats_string = ', '.join(['A@%d=%.4f' % (j + 1, ats[j]) for j in range(len(ats))])
        print('For "%s/%s": overall %s. AUC=%.6f.' % (
            path,
            model,
            ats_string,
            auc),
        flush=True)
