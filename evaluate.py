#!/usr/bin/env python

import csv
import glob
import pickle
import random
import re
import sys
import numpy as np

RANDOM_SEED = 228
random.seed(RANDOM_SEED)

if not len(sys.argv) > 1:
    print('Usage: %s model' % (sys.argv[0]), file=sys.stderr)
    sys.exit(1)

MODEL = sys.argv[1]

with np.load('train.npz') as data:
    X_all_train, Y_all_train = data['X_all_train'], data['Y_all_train']
    c1_train  = data['cd1_train'][0]
    c5_train  = data['cd5_train'][0]
    c10_train = data['cd10_train'][0]

with np.load('test.npz') as data:
    X_all_test, Y_all_test = data['X_all_test'], data['Y_all_test']
    c1_test  = data['cd1_test'][0]
    c5_test  = data['cd5_test'][0]
    c10_test = data['cd10_test'][0]

subsumptions_test = []

with open('subsumptions-test.txt') as f:
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        subsumptions_test.append((row[0], row[1]))

kmeans = pickle.load(open('kmeans.pickle', 'rb'))
print('The number of clusters is %d.' % (kmeans.n_clusters))

clusters_train = kmeans.predict(Y_all_train - X_all_train)
clusters_test  = kmeans.predict(Y_all_test  - X_all_test)

W = [None] * kmeans.n_clusters

CLUSTER_REGEXP = re.compile('W-(?P<cluster>\d+)\.txt$')

for path in glob.glob('%s.W-*.txt' % (MODEL)):
    cluster = int(CLUSTER_REGEXP.search(path).group('cluster')) - 1
    print('Loading "%s" as the cluster %d.' % (path, cluster))
    W[cluster] = np.loadtxt(path)

def cosine(v1, v2):
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return 0. if np.isnan(similarity) else similarity

measures1, measures5, measures10 = {}, {}, {}

for i, (hyponym, hypernym) in enumerate(subsumptions_test):
    cluster   = clusters_test[i]

    x_example        = np.ones((1, X_all_test.shape[1] + 1))
    x_example[:, 1:] = X_all_test[i]
    y_example        = Y_all_test[i]
    y_hat            = x_example.dot(W[cluster]).reshape(X_all_test.shape[1],)

    cosine_hat = cosine(y_example, y_hat)

    measures1[(hyponym, hypernym)]  = 1. if cosine_hat > c1_test  else 0.
    measures5[(hyponym, hypernym)]  = 1. if cosine_hat > c5_test  else 0.
    measures10[(hyponym, hypernym)] = 1. if cosine_hat > c10_test else 0.

    if (i + 1) % 100 == 0:
        print('%d examples out of %d done for "%s": A@1 is %.6f, A@5 is %.6f and A@10 is %.6f.' % (i + 1,
            len(subsumptions_test), MODEL,
            sum(measures1.values())  / len(subsumptions_test),
            sum(measures5.values())  / len(subsumptions_test),
            sum(measures10.values()) / len(subsumptions_test)), file=sys.stderr)

print('Overall A@1 is %.4f, A@5 is %.4f and A@10 is %.4f.' % (
    sum(measures1.values())  / len(subsumptions_test),
    sum(measures5.values())  / len(subsumptions_test),
    sum(measures10.values()) / len(subsumptions_test)))
