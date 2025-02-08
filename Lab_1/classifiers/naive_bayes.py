from collections import defaultdict

import numpy as np

class NaiveBayesClassifier:
    def __init__(self, laplace_smoothing_factor=0):
        self.laplace_smoothing_factor = laplace_smoothing_factor
        self.classes = []
        self.classes_probabilities = {} # [P(c_1), P(c_2), ...]
        self.features_probabilities = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) # {f1: {f1_val: {c1: P(f1|c)}}}

    def fit(self, X, y):
        self.__validate_fit_values(X, y)

        self.__init_classes(y)
        self.__init_classes_probabilities(y)
        self.__init_features_probabilities(X, y)

    def __validate_fit_values(self, X, y):
        data_rows_count = X.shape[0]
        labels_rows_count = y.shape[0]

        if data_rows_count != labels_rows_count:
            raise ValueError('Dimension mismatch with data and label rows')

    def __init_classes(self, y):
        self.classes = np.unique(y)

    def __init_classes_probabilities(self, y):
        for c in self.classes:
            self.classes_probabilities[c] = np.sum(y == c) / len(y)

    def __init_features_probabilities(self, X, y):
        for feature_idx in range(X.shape[1]):
            f_unique_vals = np.unique(X[:, feature_idx])

            for c in self.classes:
                cur_feature_subset = X[y == c, feature_idx]
                f_value_counts_map = {f_val: np.sum(cur_feature_subset == f_val) for f_val in f_unique_vals}
                total_count = len(cur_feature_subset)

                for f_val in f_unique_vals:
                    f_val_count_in_class = f_value_counts_map.get(f_val, 0)
                    self.features_probabilities[feature_idx][f_val][c] \
                        = (f_val_count_in_class + self.laplace_smoothing_factor) / (total_count + self.laplace_smoothing_factor * len(f_unique_vals))

    def predict(self, X):
        predictions = []

        for sample in X:
            final_class_probs = {}

            for c in self.classes:
                cur_class_prob = np.log(self.classes_probabilities[c])

                for feature_idx, feature_val in enumerate(sample):
                    cur_class_prob += np.log(self.features_probabilities[feature_idx][feature_val][c])

                final_class_probs[c] = cur_class_prob

            predictions.append(max(final_class_probs, key=final_class_probs.get))

        return predictions

