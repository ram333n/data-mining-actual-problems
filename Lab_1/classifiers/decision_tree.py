from collections import Counter

import numpy as np

class TreeNode:
    def __init__(self, feature_idx, label, is_leaf):
        self.feature_idx = feature_idx
        self.is_leaf = is_leaf
        self.label = label
        self.children = {}

    @staticmethod
    def create_leaf(label):
        return TreeNode( None, label, True)

    @staticmethod
    def create_decision_node(feature_idx):
        return TreeNode(feature_idx, None, False)


class DecisionTreeClassifier:
    def __init__(self):
        self.root = None
        self.possible_values_by_feature_idx = {}

    def fit(self, X, y):
        self.__validate_fit_values(X, y)
        self.__init_possible_values_for_features(X)
        self.__build_tree(X, y)

    def __validate_fit_values(self, X, y):
        data_rows_count = X.shape[0]
        labels_rows_count = y.shape[0]

        if data_rows_count != labels_rows_count:
            raise ValueError('Dimension mismatch with data and label rows')

    def __init_possible_values_for_features(self, X):
        for f_idx in range(X.shape[1]):
            self.possible_values_by_feature_idx[f_idx] = np.unique(X[:, f_idx])

    def __build_tree(self, X, y):
        self.root = self.__build_tree_internal(X, y, list(range(X.shape[1])))

    def __build_tree_internal(self, X, y, features_indices):
        if len(y) == 0:
            return None

        if len(set(y)) == 1:
            return TreeNode.create_leaf(y[0])

        if len(features_indices) == 0:
            return TreeNode.create_leaf(Counter(y).most_common(1)[0][0])

        best_feature_idx = self.__best_feature_idx(X, y, features_indices)
        node = TreeNode.create_decision_node(best_feature_idx)

        for f_val in self.possible_values_by_feature_idx[best_feature_idx]:
            samples_with_f_val_idx = X[:, best_feature_idx] == f_val
            X_subset = X[samples_with_f_val_idx, :]
            features_to_process = [val for val in features_indices if val != best_feature_idx]
            subtree = self.__build_tree_internal(X_subset, y[samples_with_f_val_idx], features_to_process)
            node.children[f_val] = subtree

        return node

    def __best_feature_idx(self, X, y, features_indices):
        return np.argmax([
            (self.__information_gain(X, y, i) if i in features_indices else -1) for i in range(X.shape[1])
        ])

    def __information_gain(self, X, y, feature_idx_to_split):
        f_values, counts = np.unique(X[:, feature_idx_to_split], return_counts=True)
        samples_count = np.sum(counts)
        total_entropy = self.__entropy(y)
        weighted_entropy = 0

        for i in range(len(f_values)):
            samples_with_specific_y_val = y[X[:, feature_idx_to_split] == f_values[i]]
            weighted_entropy += (counts[i] / samples_count) * self.__entropy(samples_with_specific_y_val)

        return total_entropy - weighted_entropy

    def __entropy(self, y):
        samples_count = y.shape[0]
        values, counts = np.unique(y, return_counts=True)
        probabilities = counts / samples_count

        return -np.sum([p * np.log2(p) for p in probabilities])

    def predict(self, X):
        return [self.__predict_sample(sample) for sample in X]

    def __predict_sample(self, sample):
        return self.__predict_sample_internal(sample, self.root)

    def __predict_sample_internal(self, sample, node):
        if node.is_leaf:
            return node.label
        if sample[node.feature_idx] in node.children:
            return self.__predict_sample_internal(sample, node.children[sample[node.feature_idx]])

        return Counter([child.label for child in node.children.values() if child.is_leaf]).most_common(1)[0][0]

    def print(self):
        self.__print_tree(self.root, 0)

    def __print_tree(self, node, depth):
        if node is None:
            return

        if node.is_leaf:
            print("  " * depth + f"Leaf({node.label})")
            return

        print("  " * depth + f"Feature: {node.feature_idx}")
        for value, child in node.children.items():
            print("  " * (depth + 1) + f"{node.feature_idx} -> {value}")
            self.__print_tree(child, depth + 2)