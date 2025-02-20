import numpy as np
import pandas as pd

from classifiers.decision_tree import DecisionTreeClassifier
from classifiers.knn import KNNClassifier
from classifiers.naive_bayes import NaiveBayesClassifier
from classifiers.one_rule import OneRuleClassifier


def read_dataset(filename):
    df = pd.read_csv(filename)
    f_names = list(filter(lambda c: c[0] == 'f', df.columns))

    return df[f_names].to_numpy(), df['target'].to_numpy()

def test_naive_bayes():
    X, y = read_dataset('./data/input_0.csv')
    bayes_classifier = NaiveBayesClassifier(0.1)
    bayes_classifier.fit(X, y)

    X_test = np.array([
        [2, 1, 1, 1]
    ])
    print(f'Predicted class: {bayes_classifier.predict(X_test)}')

def test_knn():
    X, y = read_dataset('./data/input_0.csv')
    knn_classifier = KNNClassifier(3, weighted_voting=False)
    knn_classifier.fit(X, y)

    X_test = np.array([
        [2, 1, 1, 1]
    ])
    print(f'Predicted class: {knn_classifier.predict(X_test)}')

def test_one_rule():
    X, y = read_dataset('./data/input_0.csv')
    one_rule_classifier = OneRuleClassifier()
    one_rule_classifier.fit(X, y)

    X_test = np.array([
        [2, 1, 1, 1],
    ])
    print(f'Predicted class: {one_rule_classifier.predict(X_test)}')

def test_decision_tree():
    X, y = read_dataset('./data/input_0.csv')
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(X, y)

    X_test = np.array([
        [2, 1, 1, 1],
    ])
    print(f'Predicted class: {decision_tree_classifier.predict(X_test)}')

if __name__ == '__main__':
    test_decision_tree()
