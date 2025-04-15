import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-8, centroids=None, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = centroids
        self.labels = []
        self.random_state = random_state

    def fit(self, X):
        self.centroids = self.__init_centroids_if_empty(X)
        self.labels = self.__assign_labels(X, self.centroids)

        for i in range(self.max_iter):
            old_centroids = self.centroids
            self.centroids = self.__update_centroids(X, self.labels)
            self.labels = self.__assign_labels(X, self.centroids)

            if self.__is_converged(old_centroids, self.centroids):
                break

    def __init_centroids_if_empty(self, X):
        if self.centroids is not None:
            return self.centroids

        randomizer = np.random.RandomState(self.random_state)

        return X[randomizer.choice(X.shape[0], self.n_clusters, replace=False)]

    def __assign_labels(self, X, centroids):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        return np.argmin(distances, axis=1)

    def __update_centroids(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def __is_converged(self, old_centroids, new_centroids):
        return np.max(np.linalg.norm(new_centroids - old_centroids, axis=1)) < self.tol