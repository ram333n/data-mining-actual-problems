import numpy as np

class KMedoids:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.medoids_idx = []
        self.labels = []
        self.random_state = random_state
        self.medoids = []

    def fit(self, X):
        self.__init_medoids(X)

        for _ in range(self.max_iter):
            self.labels = self.__assign_labels(X)
            new_medoids_idx = self.__update_medoids(X)

            if self.__is_converged(X, new_medoids_idx):
                break

            self.medoids_idx = new_medoids_idx

        self.medoids = X[self.medoids_idx]

    def __init_medoids(self, X):
        np.random.seed(self.random_state)
        self.medoids_idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)

    def __assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - X[self.medoids_idx], axis=2)

        return np.argmin(distances, axis=1)

    def __update_medoids(self, X):
        new_medoids = []

        for cluster in range(self.n_clusters):
            cluster_points_idx = np.where(self.labels == cluster)[0]
            costs = []

            for medoid_candidate_idx in cluster_points_idx:
                cost = np.sum([np.linalg.norm(X[medoid_candidate_idx] - X[p_idx]) for p_idx in cluster_points_idx])
                costs.append(cost)

            best_medoid = cluster_points_idx[np.argmin(costs)]
            new_medoids.append(best_medoid)

        return new_medoids

    def __is_converged(self, X, new_medoids_idx):
        return np.max(np.linalg.norm(X[new_medoids_idx] - X[self.medoids_idx], axis=1)) < self.tol
