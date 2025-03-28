import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, centroids=None, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = centroids
        self.labels = []
        self.random_state = random_state

    def fit(self, X):
        self.__init_centroids_if_empty(X)

        for _ in range(self.max_iter):
            self.labels = self.__assign_labels(X)
            new_centroids = self.__update_centroids(X)

            if self.__is_converged(new_centroids):
                break

            self.centroids = new_centroids

    def __init_centroids_if_empty(self, X):
        if self.centroids is not None:
            return

        np.random.seed(self.random_state)
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

    def __assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

        return np.argmin(distances, axis=1)

    def __update_centroids(self, X):
        return np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def __is_converged(self, new_centroids):
        return np.max(np.linalg.norm(new_centroids - self.centroids, axis=1)) < self.tol