from sklearn.datasets import make_blobs
from clustering.kmeans import KMeans
from util.plotting import PlotUtils


def main():
    X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    print('Cluster Assignments:', kmeans.labels)
    print('Final Centroids:', kmeans.centroids)

    PlotUtils.plot('K-means', X, kmeans.labels, kmeans.centroids)

if __name__ == '__main__':
    main()