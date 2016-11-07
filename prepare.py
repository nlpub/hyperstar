#!/usr/bin/env python

import argparse
import csv
import random
from gensim.models.word2vec import Word2Vec
from collections import defaultdict
import numpy as np
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

parser = argparse.ArgumentParser(description='Preparation.')
parser.add_argument('--w2v',  default='all.norm-sz100-w10-cb0-it1-min100.w2v', nargs='?', help='Path to the word2vec model.')
parser.add_argument('--seed', default=228, type=int, nargs='?', help='Random seed.')
args = vars(parser.parse_args())

RANDOM_SEED = args['seed']
random.seed(RANDOM_SEED)

w2v = Word2Vec.load_word2vec_format(args['w2v'], binary=True, unicode_errors='ignore')
w2v.init_sims(replace=True)
print('Using %d word2vec dimensions from "%s".' % (w2v.layer1_size, args['w2v']))

def read_subsumptions(filename):
    subsumptions = []

    with open(filename) as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in reader:
            subsumptions.append((row[0], row[1]))

    return subsumptions

def read_synonyms(filename):
    synonyms = defaultdict(lambda: list())

    with open(filename) as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in reader:
            for word in row[1].split(','):
                synonyms[row[0]].append(word)

    return synonyms

subsumptions_train      = read_subsumptions('subsumptions-train.txt')
subsumptions_validation = read_subsumptions('subsumptions-validation.txt')
subsumptions_test       = read_subsumptions('subsumptions-test.txt')
synonyms                = read_synonyms('synonyms.txt')

def compute_X(subsumptions, add_synonyms=True):
    X_index, X_all = [], []

    for hyponym, hypernym in subsumptions:
        offset        = len(X_all)
        word_synonyms = [hyponym] + synonyms[hyponym] if add_synonyms else [hyponym]

        X_index.append([offset, len(word_synonyms)])

        for synonym in word_synonyms:
            X_all.append(w2v[synonym])

    return (np.array(X_index, dtype='int32'), np.array(X_all))

def compute_XYZ(X_index, shuffle=False):
    XYZ = []

    for i, (offset, length) in enumerate(X_index):
        for j in range(offset, offset + length):
            # X = offset, Y = i, Z = j
            XYZ.append((offset, i, j))

    # only the training data need to be shuffled
    if shuffle:
        random.shuffle(XYZ)

    return np.array(XYZ, dtype='int32')

X_index_train,      X_all_train      = compute_X(subsumptions_train, add_synonyms=False)
X_index_validation, X_all_validation = compute_X(subsumptions_validation, add_synonyms=False)
X_index_test,       X_all_test       = compute_X(subsumptions_test, add_synonyms=False)

XYZ_train      = compute_XYZ(X_index_train, shuffle=True)
XYZ_validation = compute_XYZ(X_index_validation)
XYZ_test       = compute_XYZ(X_index_test)

Y_all_train      = np.array([w2v[w] for _, w in subsumptions_train])
Y_all_validation = np.array([w2v[w] for _, w in subsumptions_validation])
Y_all_test       = np.array([w2v[w] for _, w in subsumptions_test])

np.savez_compressed('train.npz',      X_index=X_index_train,
                                      X_all=X_all_train,
                                      Y_all=Y_all_train,
                                      XYZ=XYZ_train)

np.savez_compressed('validation.npz', X_index=X_index_validation,
                                      X_all=X_all_validation,
                                      Y_all=Y_all_validation,
                                      XYZ=XYZ_validation)

np.savez_compressed('test.npz',       X_index=X_index_test,
                                      X_all=X_all_test,
                                      Y_all=Y_all_test,
                                      XYZ=XYZ_test)

print('I have %d train, %d validation and %d test examples.' % (
    XYZ_train.shape[0],
    XYZ_validation.shape[0],
    XYZ_test.shape[0])
)
