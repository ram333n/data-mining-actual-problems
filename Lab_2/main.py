from sklearn.datasets import make_blobs
from clustering.kmeans import KMeans
from clustering.kmedoids import KMedoids
from util.plotting import PlotUtils

def test_k_means():
    X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    print('Cluster Assignments:', kmeans.labels)
    print('Final Centroids:', kmeans.centroids)

    PlotUtils.plot('K-means', X, kmeans.labels, kmeans.centroids)

def test_k_medoids():
    X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

    kmedoids = KMedoids(n_clusters=3)
    kmedoids.fit(X)

    print('Cluster Assignments:', kmedoids.labels)
    print('Final Medoids:', kmedoids.medoids)

    PlotUtils.plot('K-medoids', X, kmedoids.labels, kmedoids.medoids)

def main():
    test_k_medoids()

if __name__ == '__main__':
    main()