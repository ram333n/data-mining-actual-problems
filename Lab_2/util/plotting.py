import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

from clustering.dbscan import DBSCAN


class PlotUtils:
    cluster_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', '']

    @staticmethod
    def plot(description, X, clusters, centroids=None):
        plt.figure(figsize=(8, 8))

        for i in range(X.shape[0]):
            plt.scatter(X[i, 0], X[i, 1], color=PlotUtils.cluster_colors[clusters[i]])

        if centroids is not None:
            plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', label='Centroid')
            plt.legend()

        plt.title(description)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_dbscan(description, X, clusters):
        plt.figure(figsize=(8, 8))

        for i in range(X.shape[0]):
            if clusters[i] != DBSCAN.NOISE_LABEL:
                plt.scatter(X[i, 0], X[i, 1],
                            color=PlotUtils.cluster_colors[clusters[i]])

        noise_mask = clusters == DBSCAN.NOISE_LABEL
        plt.scatter(X[noise_mask, 0], X[noise_mask, 1], color='black', marker='x', label="Noise")

        plt.title(description)
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_dendrogram(description, agg_clustering):
        pass
        # plt.figure(figsize=(10, 5))
        # dendrogram(agg_clustering.linkage_matrix)
        # plt.title("Agglomerative Clustering Dendrogram")
        # plt.xlabel("Data Point Index")
        # plt.ylabel("Distance")
        # plt.show()