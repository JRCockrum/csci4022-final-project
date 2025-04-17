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
            raise RuntimeError(f"Unexpected error when looking up index {index}: {e}") from e