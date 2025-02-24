import numpy as np


class OneRuleClassifier:
    def __init__(self, verbose):
        self.best_feature = None
        self.rules = None
        self.verbose = verbose

    def fit(self, X, y):
        self.__validate_fit_values(X, y)

        min_error = 1
        best_feature_idx = None
        best_rules = None

        for f_idx in range(X.shape[1]):
            f_column = X[:, f_idx]
            cur_rules = self.__create_rules(f_column, y)
            cur_error = self.__evaluate_error(f_column, y, cur_rules)

            if cur_error < min_error:
                min_error = cur_error
                best_feature_idx = f_idx
                best_rules = cur_rules

        if self.verbose:
            print(f'Best feature idx: {best_feature_idx}, error rate: {min_error}')

        self.best_feature = best_feature_idx
        self.rules = best_rules

    def __validate_fit_values(self, X, y):
        data_rows_count = X.shape[0]
        labels_rows_count = y.shape[0]

        if data_rows_count != labels_rows_count:
            raise ValueError('Dimension mismatch with data and label rows')

    def __create_rules(self, f_column, y):
        rules = {}

        for f_val in np.unique(f_column):
            most_frequent_class = self.__find_the_most_frequent_class(f_column, f_val, y)
            rules[f_val] = most_frequent_class

        return rules

    def __find_the_most_frequent_class(self, f_column, f_val, y):
        values, counts = np.unique(y[f_column == f_val], return_counts=True)
        return values[np.argmax(counts)]

    def __evaluate_error(self, f_column, y, rules):
        predictions = np.array([rules[x] for x in f_column])
        return np.sum(predictions != y) / y.shape[0]

    def predict(self, X):
        f_column = X[:, self.best_feature]

        return np.array([self.rules[f_val] for f_val in f_column])

