import numpy as np


class DecisionTreeClassifier:
    def __init__(self):
        pass

    def __validate_fit_values(self, X, y):
        data_rows_count = X.shape[0]
        labels_rows_count = y.shape[0]

        if data_rows_count != labels_rows_count:
            raise ValueError('Dimension mismatch with data and label rows')

    def fit(self, X, y):
        self.__validate_fit_values(X, y)

        print(self.__entropy(y))

    def __entropy(self, y):
        samples_count = y.shape[0]
        values, counts = np.unique(y, return_counts=True)
        probabilities = counts / samples_count

        return -np.sum([p * np.log2(p) for p in probabilities])

    def __information_gain(self, X, y, feature):
        pass

    def predict(self, X):
        pass

