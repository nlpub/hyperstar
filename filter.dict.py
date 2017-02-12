#!/usr/bin/env python

import argparse
import csv
import random
from gensim.models.word2vec import Word2Vec
from collections import defaultdict
import numpy as np
import pandas as pd
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

parser = argparse.ArgumentParser(description='Preparation.')
parser.add_argument('--w2v',  default='all.norm-sz100-w10-cb0-it1-min100.w2v', nargs='?', help='Path to the word2vec model.')
args = vars(parser.parse_args())

w2v = Word2Vec.load_word2vec_format(args['w2v'], binary=True, unicode_errors='ignore')
print('Using %d word2vec dimensions from "%s".' % (w2v.layer1_size, args['w2v']))

for fname in ['subsumptions-train.txt','subsumptions-test.txt','subsumptions-validation.txt','synonyms.txt']:
    df=pd.read_csv(fname,sep='\t',header=None)
    mask = df.apply(lambda x:x[0] in w2v and x[1] in w2v, axis=1)
    print('%s: %d/%d rows contain oov-words' % (fname, (~mask).sum(), len(mask)))
    df[mask].to_csv(fname,sep='\t',header=None,index=False)

