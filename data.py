import random
import numpy as np

class Data:
    def __init__(self, cluster, clusters_train, clusters_test,
                       X_all_train, Y_all_train, Z_index_train, Z_all_train,
                       X_all_test,  Y_all_test,  Z_index_test,  Z_all_test):
        self.cluster = cluster

        self.subsumptions_train = [i for i, c in enumerate(clusters_train) if c == cluster]
        self.subsumptions_test  = [i for i, c in enumerate(clusters_test)  if c == cluster]

        self.X_train        = np.ones((len(self.subsumptions_train), X_all_train.shape[1] + 1))
        self.X_train[:, 1:] = X_all_train[self.subsumptions_train]
        self.Y_train        = Y_all_train[self.subsumptions_train]
        self.Z_index_train  = Z_index_train[self.subsumptions_train]
        self.Z_all_train    = Z_all_train

        self.X_test         = np.ones((len(self.subsumptions_test),  X_all_test.shape[1] + 1))
        self.X_test[:, 1:]  = X_all_test[self.subsumptions_test]
        self.Y_test         = Y_all_test[self.subsumptions_test]
        self.Z_index_test   = Z_index_test[self.subsumptions_test]
        self.Z_all_test     = Z_all_test

    def Z_train(self, batch=None):
        batch = range(len(self.subsumptions_train)) if batch is None else batch
        substitutions  = [random.choice(range(self.Z_index_train[i][0], self.Z_index_train[i][0] + self.Z_index_train[i][1])) for i in batch]
        Z_batch        = np.ones((len(batch), self.X_train.shape[1]))
        Z_batch[:, 1:] = np.array([self.Z_all_train[i] for i in substitutions])
        return Z_batch

    def Z_test(self, batch=None):
        batch = range(len(self.subsumptions_test)) if batch is None else batch
        substitutions  = [random.choice(range(self.Z_index_test[i][0], self.Z_index_test[i][0] + self.Z_index_test[i][1])) for i in batch]
        Z_batch        = np.ones((len(batch), self.X_test.shape[1]))
        Z_batch[:, 1:] = np.array([self.Z_all_test[i] for i in substitutions])
        return Z_batch
