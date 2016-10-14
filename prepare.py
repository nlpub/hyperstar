#!/usr/bin/env python

import argparse
import csv
import random
from collections import defaultdict
from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.cross_validation import train_test_split

parser = argparse.ArgumentParser(description='Evaluation.')
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
            hyponym, hypernym = row[2], row[3]
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
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f, dialect='excel-tab', lineterminator='\n')
        for pair in subsumptions:
            writer.writerow(pair)

write_subsumptions(subsumptions_train,      'subsumptions-train.txt')
write_subsumptions(subsumptions_validation, 'subsumptions-validation.txt')
write_subsumptions(subsumptions_test,       'subsumptions-test.txt')

with open('synonyms.txt', 'w', newline='') as f:
    writer = csv.writer(f, dialect='excel-tab', lineterminator='\n')
    for word, words in synonyms.items():
        writer.writerow((word, ','.join(words)))

def compute_Z(subsumptions):
    Z_index, Z_all = [], []

    for hyponym, hypernym in subsumptions:
        x_index       = len(Z_all)
        word_synonyms = [hyponym] + synonyms[hyponym]

        Z_index.append([x_index, len(word_synonyms)])

        for synonym in word_synonyms:
            Z_all.append(w2v[synonym])

    return (np.array(Z_index, dtype='int32'), np.array(Z_all))

Z_index_train,      Z_all_train      = compute_Z(subsumptions_train)
Z_index_validation, Z_all_validation = compute_Z(subsumptions_validation)
Z_index_test,       Z_all_test       = compute_Z(subsumptions_test)

Y_all_train      = np.array([w2v[w] for _, w in subsumptions_train])
Y_all_validation = np.array([w2v[w] for _, w in subsumptions_validation])
Y_all_test       = np.array([w2v[w] for _, w in subsumptions_test])

np.savez_compressed('train.npz',      Y_all=Y_all_train,
                                      Z_index=Z_index_train,
                                      Z_all=Z_all_train)

np.savez_compressed('validation.npz', Y_all=Y_all_validation,
                                      Z_index=Z_index_validation,
                                      Z_all=Z_all_validation)

np.savez_compressed('test.npz',       Y_all=Y_all_test,
                                      Z_index=Z_index_test,
                                      Z_all=Z_all_test)

print('I have %d train, %d validation and %d test examples.' % (Y_all_train.shape[0], Y_all_validation.shape[0], Y_all_test.shape[0]))
print('Also, I have %d train, %d validation and %d test synonyms.' % (
    Z_all_train.shape[0]      - Y_all_train.shape[0],
    Z_all_validation.shape[0] - Y_all_validation.shape[0],
    Z_all_test.shape[0]       - Y_all_test.shape[0])
)
