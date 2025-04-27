from cluster import Cluster
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self, k: int, data: pd.DataFrame, numeric_cols: list, cat_cols: list, tol=0.05, a=0.995):
        self._data = data
        self._clusters = [Cluster(self) for _ in range(k)]
        self._numeric_cols = numeric_cols
        self._cat_cols = cat_cols
        self._tol = tol
        self._a = a

        #Init Clusters with random datapoint
        random_indices = random.sample(range(len(data)), k)
        for cluster, idx in zip(self._clusters, random_indices):
            cluster.add_row(idx)


    def get_clusters(self):
        return self._clusters

    def lookup_numeric(self, index):
        try:
            return self._data.loc[index, self._numeric_cols].values
        except IndexError as e:
            raise IndexError(f"Index {index} is out of bounds for the cluster data.") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error when looking up index {index}: {e}") from e
        
    def lookup_cat(self, index):
        try:
            return self._data.loc[index, self._cat_cols].values
        except IndexError as e:
            raise IndexError(f"Index {index} is out of bounds for the cluster data.") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error when looking up index {index}: {e}") from e
        
    def run_kmeans(self):
        prev_error = np.inf
        while True:
            for row_num in range(len(self._data)):
                num_centroids = [clust.get_num_centroid() for clust in self._clusters]
                cat_centroids = [clust.get_cat_centroid() for clust in self._clusters]

                num_row_vals = self.lookup_numeric(row_num)
                cat_row_vals = self.lookup_cat(row_num)

                closest_cluster = 0
                closest_dist = self.dist(num_centroids[0], num_row_vals, cat_centroids[0], cat_row_vals)

                for centroid_num in range(len(self._clusters)):
                    d = self.dist(num_centroids[centroid_num], num_row_vals, cat_centroids[centroid_num], cat_row_vals)
                    if d < closest_dist:
                        closest_cluster = centroid_num
                        closest_dist = d

                for clust in self._clusters:
                    if clust.has_row(row_num):
                        clust.drop_row(row_num)
                self._clusters[closest_cluster].add_row(row_num)

            error = self.calculate_error()
            print(error)
            if abs(prev_error - error) <= self._tol:
                self.show_clusters()
                return
            else:
                prev_error = error

        
    def dist(self, num_x, num_y, cat_x, cat_y):
        return (1-self._a)*self._euc_dist(num_x,num_y) + self._a*self._jac_dist(cat_x,cat_y)
    
    def _euc_dist(self, x, y):
        return np.sqrt(sum((x_i-y_j)**2 for x_i, y_j in zip(x, y)))

    def _jac_dist(self, x, y):
        x = set(x)
        y = set(y)
        sim = len(x & y) / len(x | y)
        return 1-sim

    def calculate_error(self):
        error = 0
        for cluster in self._clusters:
            error += cluster.get_error()
        return error
    
    def show_clusters(self):
        # TODO add ability to selext axes + title + labels
        plt.figure(figsize=(8, 6))
        COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

        for i, cluster in enumerate(self._clusters):
            x = [self.lookup_numeric(idx)[1] for idx in cluster.get_data()]
            y = [self.lookup_numeric(idx)[0] for idx in cluster.get_data()]
            plt.scatter(x, y, color=COLORS[i])
        plt.show()
        