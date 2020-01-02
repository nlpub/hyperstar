__author__ = 'Dmitry Ustalov'

import random


class Data:
    def __init__(self, cluster, clusters_train, clusters_test,
                 X_index_train, Y_all_train, Z_all_train,
                 X_index_test, Y_all_test, Z_all_test):
        self.cluster = cluster

        self.subsumptions_train = [i for i, c in enumerate(clusters_train) if c == self.cluster]
        self.subsumptions_test = [i for i, c in enumerate(clusters_test) if c == self.cluster]

        self.X_index_train, self.Y_all_train, self.Z_all_train = X_index_train, Y_all_train, Z_all_train
        self.X_index_test, self.Y_all_test, self.Z_all_test = X_index_test, Y_all_test, Z_all_test

        self.X_train, self.Y_train, self.Z_train = self.fetch(self.subsumptions_train, self.X_index_train,
                                                              self.Y_all_train, self.Z_all_train)
        self.X_test, self.Y_test, self.Z_test = self.fetch(self.subsumptions_test, self.X_index_test, self.Y_all_test,
                                                           self.Z_all_test)

    def train_shuffle(self):
        subsumptions = self.subsumptions_train[:]
        random.shuffle(subsumptions)
        return self.fetch(subsumptions, self.X_index_train, self.Y_all_train, self.Z_all_train)

    def fetch(self, subsumptions, X_index, Y_all, Z_all):
        X = Z_all[X_index[subsumptions, 0]]
        Y = Y_all[subsumptions]
        Z = self.sample_Z(subsumptions, X_index, Z_all)
        assert X.shape[0] == Y.shape[0] == Z.shape[0]
        return (X, Y, Z)

    def sample_Z(self, subsumptions, X_index, Z_all):
        indices = [random.choice(range(X_index[i][0], X_index[i][0] + X_index[i][1])) for i in subsumptions]
        assert len(indices) == len(subsumptions)
        return Z_all[indices]
