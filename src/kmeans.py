from cluster import Cluster
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