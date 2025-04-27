from cluster import Cluster
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self, k: int, data: pd.DataFrame, tol=0.05):
        self._data = data
        self._clusters = [Cluster(self) for _ in range(k)]
        self._tol = tol
        self._error = 0


    def get_clusters(self):
        return self._clusters

    def lookup_index(self, index):
        try:
            return self._data.iloc[index, :]
        except IndexError as e:
            raise IndexError(f"Index {index} is out of bounds for the cluster data.") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error when looking up index {index}: {e}") from e
        
    def run_kmeans(self):
        # TODO init cluster with rand datapoints
        prev_error = np.inf
        while True:
            for row_num in range(len(self._data)):
                centroids = [clust.get_centroid() for clust in self._clusters]
                row_vals = self.lookup_index(row_num)
                closest_centroid = 0
                closest_dist = self.dist(centroids[0], row_vals)

                for centroid_num in range(len(centroids)):
                    d = self.dist(centroids[centroid_num], row_vals)
                    if d < closest_dist:
                        closest_centroid = centroid_num
                        closest_dist = d

                for clust in self._clusters:
                    if clust.has_row(row_num):
                        clust.drop_row(row_num)
                self._clusters[closest_centroid].add_row(row_num)


            error = self.calculate_error()
            if abs(prev_error - error) <= self._tol:
                self.show_clusters()
                return
            else:
                prev_error = error

        
    def dist(self, x, y):
        return np.sqrt(sum((x_i-y_j)**2 for x_i, y_j in zip(x, y)))

    def calculate_error(self):
        error = 0
        for cluster in self._clusters:
            error += cluster.get_error()
        return error
    
    def show_clusters(self):
        # TODO add ability to selext axes + title + labels
        plt.figure(figsize=(8, 6))
        for cluster in self._clusters:
            x = [self.lookup_index(idx).values[0] for idx in cluster.get_data()]
            y = [self.lookup_index(idx).values[1] for idx in cluster.get_data()]
            plt.scatter(x, y)
        plt.show()
        