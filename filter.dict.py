#!/usr/bin/env python3

__author__ = 'Nikolay Arefyev'

import argparse

import pandas as pd
from gensim.models.word2vec import Word2Vec

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

parser = argparse.ArgumentParser(
    description='Filter out pairs with oov-words. Rewrite input files if --rewrite, otherwise just estimate the number of oov-words in each file.')
parser.add_argument('--w2v', default='all.norm-sz100-w10-cb0-it1-min100.w2v', nargs='?',
                    help='Path to the word2vec model.')
parser.add_argument('--rewrite', action='store_true', help='Rewrite input files.')
args = vars(parser.parse_args())

w2v = Word2Vec.load_word2vec_format(args['w2v'], binary=True, unicode_errors='ignore')
print('Using %d word2vec dimensions for %d words from "%s".' % (w2v.layer1_size, len(w2v.vocab), args['w2v']))

for fname in ['subsumptions-train.txt', 'subsumptions-test.txt', 'subsumptions-validation.txt', 'synonyms.txt']:
    df = pd.read_csv(fname, sep='\t', header=None)
    mask = df.apply(lambda x: x[0] in w2v and x[1] in w2v, axis=1)
    print('%s: %d/%d rows contain oov-words' % (fname, (~mask).sum(), len(mask)))
    if args['rewrite']:
        df[mask].to_csv(fname, sep='\t', header=None, index=False, encoding='utf8')
