import pandas as pd
import random
from mode_cluster import Cluster

class KModes:
    def __init__(self, k: int, data: pd.DataFrame):
        self._data = data
        self._clusters = [Cluster(self) for _ in range(k)]

        # Random cluster init
        random_indices = random.sample(range(len(data)), k)
        for cluster, idx in zip(self._clusters, random_indices):
            cluster.add_row(idx)
            cluster.update_centroid()

        # Clustering
        while True:
            clust_changes = 0
            for row_num in range(len(self._data)):
                centroids = self.get_centroids()
                row_vals = self.get_row(row_num)

                closest_cluster = 0
                closest_dist = self._dist(centroids[0], row_vals)

                for clust_num in range(len(self._clusters)):
                    d = self._dist(centroids[clust_num], row_vals)
                    if d < closest_dist:
                        closest_cluster = clust_num
                        closest_dist = d

                if not self._clusters[closest_cluster].has_row(row_num):
                    current_cluster = next((clust for clust in self._clusters if clust.has_row(row_num)), None)
                    if current_cluster is not None:
                        current_cluster.drop_row(row_num)

                    self._clusters[closest_cluster].add_row(row_num)
                    clust_changes += 1
            
            if clust_changes == 0:
                return
            else:
                self._update_centroids()

    def get_clusters(self):
        return self._clusters
    
    def get_centroids(self):
        return [clust.get_centroid() for clust in self._clusters]

    def get_row(self, index):
        try:
            return self._data.iloc[index, :].values
        except IndexError as e:
            raise IndexError(f"Index {index} is out of bounds.") from e
        
    def _dist(self, x, y):
        diffs = 0
        for x_a, y_a in zip(x,y):
            if x_a != y_a:
                diffs += 1
        return diffs

    def _update_centroids(self):
        for clust in self._clusters:
            clust.update_centroid()


