from cluster import Cluster
import numpy as np
import pandas as pd

class Kmeans:
    def __init__(self, k: int, data: pd.DataFrame, tolerance, max_iters):
        self._data = data
        self._clusters = [Cluster(self) for _ in range(k)]
        self._tol = tolerance
        self._max_iters = max_iters


    def get_clusters(self):
        return self._clusters

    def lookup_index(self, index):
        try:
            return self._data.iloc[index, :]
        except IndexError as e:
            raise IndexError(f"Index {index} is out of bounds for the cluster data.") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error when looking up index {index}: {e}") from efrom cluster import Cluster
import pandas as pd
import random

class Cluster:
    def __init__(self, data: pd.DataFrame, k: int, tol):
        self._data = data
        self._clusters = [Cluster() for _ in range(k)]
        self._error = None
        self._tol = tol

        # Initing each cluster with random point
        for cluster in self._clusters:
            row_num = random.randint(0, self._data.shape[0]) # TODO need to make sure each rownum is different
            cluster.add_row(row_num)
            self._update_error()

    def _update_error(self):
        #TODO this computes the combined error for all clusters
        pass

    def get_error(self):
        return self._error
    
    def get_centroids(self):
        return  [cluster.get_centroid() for cluster in self._clusters]

    def run(self):
        """ 
        Loop through data
            for each data point find closest centroid 
            if point already in cluster
                remove from old cluster
            add point to new cluster
        recalculate error
        check tol
        """
        pass

    def display(self):
        #TODO print graph of clusters ( on given cols)
        pass