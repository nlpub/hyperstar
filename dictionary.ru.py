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

parser = argparse.ArgumentParser(description='Russian Dictionary.')
parser.add_argument('--w2v',  default='all.norm-sz100-w10-cb0-it1-min100.w2v', nargs='?', help='Path to the word2vec model.')
parser.add_argument('--seed', default=228, type=int, nargs='?', help='Random seed.')
args = vars(parser.parse_args())

RANDOM_SEED = args['seed']
random.seed(RANDOM_SEED)

w2v = Word2Vec.load_word2vec_format(args['w2v'], binary=True, unicode_errors='ignore')
w2v.init_sims(replace=True)
print('Using %d word2vec dimensions from "%s".' % (w2v.layer1_size, args['w2v']))

hypernyms_patterns   = defaultdict(lambda: list())
hypernyms_wiktionary = defaultdict(lambda: list())
synonyms             = defaultdict(lambda: list())

with open('pairs-isas-aa.csv') as f:
    reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        hyponym, hypernym, frequency = row['hyponym'], row['hypernym'], float(row['freq'])
        if frequency < 100:
            continue
        if hyponym in w2v and hypernym in w2v and hypernym not in hypernyms_patterns[hyponym]:
            hypernyms_patterns[hyponym].append(hypernym)

with open('all_ru_pairs_ruwikt20160210_parsed.txt') as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        hyponym, hypernym = None, None
        if row[3] == 'hypernyms':
            hyponym, hypernym = row[1], row[2]
        elif row[3] == 'hyponyms':
            hyponym, hypernym = row[2], row[1]
        elif row[3] == 'synonyms':
            if row[1] in w2v and row[2] in w2v:
                if row[2] not in synonyms[row[1]]:
                    synonyms[row[1]].append(row[2])
                if row[1] not in synonyms[row[2]]:
                    synonyms[row[2]].append(row[1])
            continue
        else:
            continue
        if hypernym not in hypernyms_wiktionary[hyponym] and hyponym in w2v and hypernym in w2v:
            hypernyms_wiktionary[hyponym].append(hypernym)

keys_wiktionary = [k for k in hypernyms_wiktionary.keys() if len(hypernyms_wiktionary[k]) > 0]

wiktionary_train, wiktionary_validation_test = train_test_split(np.arange(len(keys_wiktionary), dtype='int32'), test_size=.4, random_state=RANDOM_SEED)
wiktionary_validation, wiktionary_test = train_test_split(wiktionary_validation_test, test_size=.5, random_state=RANDOM_SEED)

hypernyms_train = {k: hypernyms_wiktionary[k] for i in wiktionary_train for k in (keys_wiktionary[i],)}

for hyponym, hypernyms in hypernyms_patterns.items():
    if hyponym in hypernyms_train:
        for hypernym in hypernyms:
            if not hypernym in hypernyms_train[hyponym]:
                hypernyms_train[hyponym].append(hypernym)

hypernyms_validation = {k: hypernyms_wiktionary[k] for i in wiktionary_validation for k in (keys_wiktionary[i],)}
hypernyms_test       = {k: hypernyms_wiktionary[k] for i in wiktionary_test       for k in (keys_wiktionary[i],)}

subsumptions_train      = [(x, y) for x, ys in hypernyms_train.items()      for y in ys]
subsumptions_validation = [(x, y) for x, ys in hypernyms_validation.items() for y in ys]
subsumptions_test       = [(x, y) for x, ys in hypernyms_test.items()       for y in ys]

def write_subsumptions(subsumptions, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f, dialect='excel-tab', lineterminator='\n')
        for pair in subsumptions:
            writer.writerow(pair)

write_subsumptions(subsumptions_train,      'subsumptions-train.txt')
write_subsumptions(subsumptions_validation, 'subsumptions-validation.txt')
write_subsumptions(subsumptions_test,       'subsumptions-test.txt')

with open('synonyms.txt', 'w') as f:
    writer = csv.writer(f, dialect='excel-tab', lineterminator='\n')
    for word, words in synonyms.items():
        writer.writerow((word, ','.join(words)))
