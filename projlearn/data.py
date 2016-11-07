import random
import numpy as np

class Data:
    def __init__(self, cluster, clusters_train, clusters_test,
                       X_all_train, Y_all_train, Z_all_train,
                       X_all_test,  Y_all_test,  Z_all_test):
        self.cluster = cluster

        self.subsumptions_train = [i for i, c in enumerate(clusters_train) if c == self.cluster]
        self.subsumptions_test  = [i for i, c in enumerate(clusters_test)  if c == self.cluster]

        self.X_train = X_all_train[self.subsumptions_train]
        self.Y_train = Y_all_train[self.subsumptions_train]
        self.Z_train = Z_all_train[self.subsumptions_train]

        self.X_test  = X_all_test[self.subsumptions_test]
        self.Y_test  = Y_all_test[self.subsumptions_test]
        self.Z_test  = Z_all_test[self.subsumptions_test]
