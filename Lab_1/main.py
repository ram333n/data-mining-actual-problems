import pandas as pd

from classifiers.naive_bayes import NaiveBayesClassifier

def read_dataset(filename):
    df = pd.read_csv(filename)
    f_names = list(filter(lambda c: c[0] == 'f', df.columns))

    return df[f_names].to_numpy(), df['target'].to_numpy()

def test_naive_bayes():
    X, y = read_dataset('./data/input_0.csv')
    bayes_classifier = NaiveBayesClassifier(0.1)
    bayes_classifier.fit(X, y)

    X_test = [
        [2, 1, 1, 1]
    ]
    print(f'Predicted class: {bayes_classifier.predict(X_test)}')


if __name__ == '__main__':
    test_naive_bayes()
