#!/usr/bin/env python

import csv
import random
from collections import defaultdict
from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.cross_validation import train_test_split

RANDOM_SEED = 228
random.seed(RANDOM_SEED)

w2v = Word2Vec.load_word2vec_format('all.norm-sz100-w10-cb0-it1-min100.w2v', binary=True, unicode_errors='ignore')
w2v.init_sims(replace=True)
e_size = w2v.layer1_size
print('Using %d word2vec dimensions.' % e_size)

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

WI_train, WI_test = train_test_split(
    np.arange(len(keys_wiktionary), dtype='int32'),
    test_size=.33,
    random_state=RANDOM_SEED)

hypernyms_train = {}

for i in WI_train:
    k = keys_wiktionary[i]
    hypernyms_train[k] = hypernyms_wiktionary[k]

for hyponym, hypernyms in hypernyms_patterns.items():
    if hyponym in hypernyms_train:
        for hypernym in hypernyms:
            if not hypernym in hypernyms_train[hyponym]:
                hypernyms_train[hyponym].append(hypernym)

hypernyms_test  = {}

for i in WI_test:
    k = keys_wiktionary[i]
    hypernyms_test[k] = hypernyms_wiktionary[k]

subsumptions_train = [(x, y) for x, ys in hypernyms_train.items() for y in ys]
subsumptions_test  = [(x, y) for x, ys in hypernyms_test.items()  for y in ys]

with open('subsumptions-train.txt', 'w', newline='') as f:
    writer = csv.writer(f, dialect='excel-tab', lineterminator='\n')
    for pair in subsumptions_train:
        writer.writerow(pair)

with open('subsumptions-test.txt', 'w', newline='') as f:
    writer = csv.writer(f, dialect='excel-tab', lineterminator='\n')
    for pair in subsumptions_test:
        writer.writerow(pair)

with open('synonyms.txt', 'w', newline='') as f:
    writer = csv.writer(f, dialect='excel-tab', lineterminator='\n')
    for word, words in synonyms.items():
        writer.writerow((word, ','.join(words)))

def cosine(v1, v2):
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return 0. if np.isnan(similarity) else similarity

X_all_train = np.array([w2v[w] for w, _ in subsumptions_train])
Y_all_train = np.array([w2v[w] for _, w in subsumptions_train])

X_all_test  = np.array([w2v[w] for w, _ in subsumptions_test])
Y_all_test  = np.array([w2v[w] for _, w in subsumptions_test])

Z_index_train, Z_all_train = [], []

for hyponym, hypernym in subsumptions_train:
    y_example = w2v[hypernym]

    word_synonyms = synonyms[hyponym] if len(synonyms[hyponym]) > 0 else [hyponym]
    Z_index_train.append([len(Z_all_train), len(word_synonyms)])
    for synonym in word_synonyms:
        Z_all_train.append(w2v[synonym])

Z_index_train = np.array(Z_index_train, dtype='int32')
Z_all_train = np.array(Z_all_train)

Z_index_test, Z_all_test = [], []

for hyponym, hypernym in subsumptions_test:
    y_example = w2v[hypernym]

    word_synonyms = synonyms[hyponym] if len(synonyms[hyponym]) > 0 else [hyponym]
    Z_index_test.append([len(Z_all_test), len(word_synonyms)])
    for synonym in word_synonyms:
        Z_all_test.append(w2v[synonym])

Z_index_test = np.array(Z_index_test, dtype='int32')
Z_all_test = np.array(Z_all_test)

np.savez_compressed('train.npz', X_all_train=X_all_train,
                                 Y_all_train=Y_all_train,
                                 Z_index_train=Z_index_train,
                                 Z_all_train=Z_all_train)

np.savez_compressed('test.npz',  X_all_test=X_all_test,
                                 Y_all_test=Y_all_test,
                                 Z_index_test=Z_index_test,
                                 Z_all_test=Z_all_test)

print('I have %d train examples and %d test examples.' % (X_all_train.shape[0], X_all_test.shape[0]))
print('Also, I have %d train synonyms and %d test synonyms.' % (Z_all_train.shape[0], Z_all_test.shape[0]))
