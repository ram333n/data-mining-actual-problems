from collections import Counter
from enum import Enum

import numpy as np


class Distance(Enum):
    EUCLIDEAN = 1


class KNNClassifier:
    def __init__(self, k, distance_metric=Distance.EUCLIDEAN, weighted_voting=True, verbose=False):
        self.k = k
        self.distance_metric = distance_metric
        self.is_weighted_voting = weighted_voting
        self.X_train = None
        self.y_train = None
        self.verbose = verbose

    def fit(self, X, y):
        self.__validate_fit_values(X, y)

        self.X_train = X
        self.y_train = y

    def __validate_fit_values(self, X, y):
        data_rows_count = X.shape[0]
        labels_rows_count = y.shape[0]

        if data_rows_count != labels_rows_count:
            raise ValueError('Dimension mismatch with data and label rows')

    def predict(self, X):
        return np.array(
            [self.__predict_sample(sample) for sample in X]
        )

    def __predict_sample(self, sample):
        distances = [self.__eval_distance(sample, x) for x in self.X_train]
        k_neighbors_indices = np.argsort(distances)[:self.k]

        if self.verbose:
            print(f'K-nearest-neighbors rows indices: {k_neighbors_indices}')

        return self.__perform_voting(distances, k_neighbors_indices)

    def __eval_distance(self, x, y):
        if self.distance_metric == Distance.EUCLIDEAN:
            return np.linalg.norm(x - y)

    def __perform_voting(self, distances, k_neighbors_indices):
        if self.is_weighted_voting:
            return self.__perform_weighted_voting(distances, k_neighbors_indices)
        else:
            return Counter([self.y_train[kn_index] for kn_index in k_neighbors_indices]).most_common(1)[0][0]

    def __perform_weighted_voting(self, distances, k_neighbors_indices):
        voting_results = {}

        for kn_index in k_neighbors_indices:
            label = self.y_train[kn_index]
            label_vote = voting_results.get(label, 0) + 1 / distances[kn_index] ** 2
            voting_results[label] = label_vote

        if self.verbose:
            print(f'Best vote: {voting_results.get(max(voting_results, key=voting_results.get))}')

        return max(voting_results, key=voting_results.get)
