import matplotlib.pyplot as plt
import numpy as np

from clustering.dbscan import DBSCAN


class PlotUtils:
    cluster_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']

    @staticmethod
    def plot(description, X, clusters, centroids=None):
        plt.figure(figsize=(8, 8))

        for i in range(X.shape[0]):
            plt.scatter(X[i, 0], X[i, 1], color=PlotUtils.cluster_colors[clusters[i]])

        if centroids is not None:
            plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x')

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
            else:
                plt.scatter(X[i, 0], X[i, 1], color='black', marker='x')

        plt.title(description)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_dendrogram(description, agg_clustering):
        # Convert the linkage matrix to the correct format and plot the dendrogram
        plt.figure(figsize=(10, 6))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index or (cluster size)')
        plt.ylabel('Distance')

        # Plot the dendrogram
        plt.imshow(np.array(agg_clustering.linkage_matrix), aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.show()
