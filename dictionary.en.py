#!/usr/bin/env python3

__author__ = 'Dmitry Ustalov'

import argparse
import csv
import os
import random
from collections import defaultdict

import numpy as np
from gensim.models.word2vec import Word2Vec

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

parser = argparse.ArgumentParser(description='LexNet-Based English Dictionary.')
parser.add_argument('--w2v', default='corpus_en.norm-sz100-w8-cb1-it1-min20.w2v', nargs='?',
                    help='Path to the word2vec model.')
parser.add_argument('--seed', default=228, type=int, nargs='?', help='Random seed.')
args = vars(parser.parse_args())

RANDOM_SEED = args['seed']
random.seed(RANDOM_SEED)

w2v = Word2Vec.load_word2vec_format(args['w2v'], binary=True, unicode_errors='ignore')
w2v.init_sims(replace=True)
print('Using %d word2vec dimensions from "%s".' % (w2v.layer1_size, args['w2v']))

positives_trusted = defaultdict(lambda: list())
negatives = defaultdict(lambda: list())

for dataset in ('K&H+N', 'BLESS', 'ROOT09', 'EVALution'):
    for part in ('train', 'val', 'test'):
        with open(os.path.join(dataset, part + '.tsv')) as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                hyponym, hypernym, relation = row[0], row[1], row[2]

                if hyponym not in w2v or hypernym not in w2v:
                    continue

                # (K&H+N, BLESS, ROOT09, EVALution)
                if relation in ('hypo', 'hyper', 'HYPER', 'IsA') and hypernym not in positives_trusted[hyponym]:
                    positives_trusted[hyponym].append(hypernym)
                elif relation in ('coord', 'Synonym'):
                    if hypernym not in negatives[hyponym]:
                        negatives[hyponym].append(hypernym)

                    if hyponym not in negatives[hypernym]:
                        negatives[hypernym].append(hyponym)

positives_untrusted = defaultdict(lambda: list())

with open('en_ps59g-rnk3-min100-nomwe-39k.csv') as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        hyponym, hypernym, frequency = row[0], row[1], float(row[2])
        if hyponym in w2v and hypernym in w2v and hypernym not in positives_untrusted[hyponym]:
            positives_untrusted[hyponym].append(hypernym)

keys_trusted = [k for k in positives_trusted.keys() if len(positives_trusted[k]) > 0]

trusted_train, trusted_validation_test = train_test_split(np.arange(len(keys_trusted), dtype='int32'), test_size=.4,
                                                          random_state=RANDOM_SEED)
trusted_validation, trusted_test = train_test_split(trusted_validation_test, test_size=.5, random_state=RANDOM_SEED)

hypernyms_train = {k: positives_trusted[k] for i in trusted_train for k in (keys_trusted[i],)}

for hyponym, hypernyms in positives_untrusted.items():
    if hyponym in hypernyms_train:
        for hypernym in hypernyms:
            if not hypernym in hypernyms_train[hyponym]:
                hypernyms_train[hyponym].append(hypernym)

hypernyms_validation = {k: positives_trusted[k] for i in trusted_validation for k in (keys_trusted[i],)}
hypernyms_test = {k: positives_trusted[k] for i in trusted_test for k in (keys_trusted[i],)}

subsumptions_train = [(x, y) for x, ys in hypernyms_train.items() for y in ys]
subsumptions_validation = [(x, y) for x, ys in hypernyms_validation.items() for y in ys]
subsumptions_test = [(x, y) for x, ys in hypernyms_test.items() for y in ys]


def write_subsumptions(subsumptions, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f, dialect='excel-tab', lineterminator='\n')
        for pair in subsumptions:
            writer.writerow(pair)


write_subsumptions(subsumptions_train, 'subsumptions-train.txt')
write_subsumptions(subsumptions_validation, 'subsumptions-validation.txt')
write_subsumptions(subsumptions_test, 'subsumptions-test.txt')

with open('synonyms.txt', 'w', newline='') as f:
    writer = csv.writer(f, dialect='excel-tab', lineterminator='\n')
    for word, words in negatives.items():
        writer.writerow((word, ','.join(words)))
