import numpy as np
from numpy import dtype


class AgglomerativeClustering:
    def __init__(self, n_clusters=3, linkage='ward'):
        self.n_clusters = n_clusters
        self.clusters_points = []
        self.linkage = linkage
        self.linkage_matrix = []
        self.labels = []

    def fit(self, X):
        n_samples = X.shape[0]
        self.clusters_points = [[i] for i in range(n_samples)]

        while len(self.clusters_points) > self.n_clusters:
            print(len(self.clusters_points))

            min_linkage = np.inf
            closest_clusters = (-1, -1)

            for i in range(len(self.clusters_points)):
                for j in range(i + 1, len(self.clusters_points)):
                    cluster_i = self.clusters_points[i]
                    cluster_j = self.clusters_points[j]
                    cur_linkage = self.__eval_linkage(cluster_i, cluster_j, X)

                    if cur_linkage < min_linkage:
                        min_linkage = cur_linkage
                        closest_clusters = (i, j)

            i, j = closest_clusters
            self.linkage_matrix.append([
                self.clusters_points[i][0],
                self.clusters_points[j][0],
                min_linkage,
                len(self.clusters_points[i]) + len(self.clusters_points[j])
            ])

            self.clusters_points[i] = self.clusters_points[i] + self.clusters_points[j]
            del self.clusters_points[j]

        self.__finalize_labels(X)

    def __distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def __eval_linkage(self, cluster_i, cluster_j, X):
        distances = [self.__distance(X[p1], X[p2]) for p1 in cluster_i for p2 in cluster_j]

        return np.min(distances) #TODO: implement other linkage methods

    def __finalize_labels(self, X):
        self.labels = np.zeros(X.shape[0], dtype=int)
        for cluster_idx, cluster in enumerate(self.clusters_points):
            for point in cluster:
                self.labels[point] = cluster_idx