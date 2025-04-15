import numpy as np


class AgglomerativeClustering:
    def __init__(self, n_clusters=3, linkage='ward'):
        self.n_clusters = n_clusters
        self.clusters = {}
        self.linkage = linkage
        self.linkage_matrix = []
        self.labels = []
        self.linkage_evaluators = {
            'single': self.__eval_single_distance,
            'complete': self.__eval_complete_distance,
            'average': self.__eval_average_distance,
            'ward': self.__eval_ward_distance
        }

    def fit(self, X):
        n_samples = X.shape[0]
        self.clusters = self.__init_clusters(X)

        while len(self.clusters.keys()) != self.n_clusters:
            cluster_i_id, cluster_j_id, dist = self.__find_closest_clusters(X)
            self.clusters = self.__merge_clusters(X, cluster_i_id, cluster_j_id, dist)

        self.labels = self.__finalize_labels(n_samples)

    def __init_clusters(self, X):
        return {sample_idx: [sample_idx] for sample_idx in range(X.shape[0])}

    def __find_closest_clusters(self, X):
        min_dist = np.inf
        closest_clusters = None

        clusters_ids = list(self.clusters.keys())

        for i, cluster_i_idx in enumerate(clusters_ids[:-1]):
            for _, cluster_j_idx in enumerate(clusters_ids[i + 1:]):
                cluster_i = self.clusters[cluster_i_idx]
                cluster_j = self.clusters[cluster_j_idx]

                dist = self.linkage_evaluators[self.linkage](X, cluster_i, cluster_j)

                if dist < min_dist:
                    min_dist = dist
                    closest_clusters = (cluster_i_idx, cluster_j_idx)

        closest_cluster_i, closest_cluster_j = closest_clusters

        return closest_cluster_i, closest_cluster_j, min_dist

    def __distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def __eval_single_distance(self, X, cluster_i, cluster_j):
        cluster_i_points = X[cluster_i]
        cluster_j_points = X[cluster_j]
        distances = np.array([self.__distance(p1, p2) for p1 in cluster_i_points for p2 in cluster_j_points])

        return np.min(distances)

    def __eval_complete_distance(self, X, cluster_i, cluster_j):
        cluster_i_points = X[cluster_i]
        cluster_j_points = X[cluster_j]
        distances = np.array([self.__distance(p1, p2) for p1 in cluster_i_points for p2 in cluster_j_points])

        return np.max(distances)

    def __eval_average_distance(self, X, cluster_i, cluster_j):
        cluster_i_points = X[cluster_i]
        cluster_j_points = X[cluster_j]
        distances = np.array([self.__distance(p1, p2) for p1 in cluster_i_points for p2 in cluster_j_points])

        return np.average(distances)

    def __eval_ward_distance(self, X, cluster_i, cluster_j):
        cluster_i_points = X[cluster_i]
        cluster_j_points = X[cluster_j]

        n_i = len(cluster_i_points)
        n_j = len(cluster_j_points)

        centroid_i = np.mean(cluster_i_points, axis=0)
        centroid_j = np.mean(cluster_j_points, axis=0)

        squared_dist = np.sum((self.__distance(centroid_i, centroid_j)) ** 2)

        return (n_i * n_j) / (n_i + n_j) * squared_dist

    def __merge_clusters(self, X, cluster_i_id, cluster_j_id, dist):
        new_clusters = {0: self.clusters[cluster_i_id] + self.clusters[cluster_j_id]}

        for cluster_id in self.clusters.keys():
            if cluster_id in [cluster_i_id, cluster_j_id]:
                continue

            new_clusters[len(new_clusters.keys())] = self.clusters[cluster_id]

        return new_clusters

    def __finalize_labels(self, n_samples):
        labels = np.zeros(n_samples, dtype=int)

        for cluster_id, cluster_pts in self.clusters.items():
            labels[cluster_pts] = cluster_id

        return labels