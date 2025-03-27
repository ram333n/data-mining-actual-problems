import matplotlib.pyplot as plt


class PlotUtils:
    cluster_colors = ['red', 'blue', 'green', 'orange', 'purple']

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
