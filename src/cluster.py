import numpy as np
import pandas as pd
# from kmeans import Kmeans

class Cluster:
    def __init__(self, kmeans):
        self._centroid = None
        self._data =  [] # list of indexes
        self._kmeans = kmeans

    def add_row(self, row):
        #TODO add check that row is not in data
        self._data.append(row)
        self._update_centroid()


    def drop_row(self, row): # row: expected to be a 1D NumPy array
        try:
            self._data.remove(row)
            self._update_centroid()
            self._update_error()
        except ValueError:
            print("Error: unable to remove row because it was not found in the cluster")
            print(f"Row: {row}")

    def _update_centroid(self):
        if not self._data:
            self._centroid = None
            return
        # Get the actual rows from the parent KMeans object
        rows = [self._kmeans.lookup_index(i).values for i in self._data]
        self._centroid = np.mean(rows, axis=0)

    def get_centroid(self):
        return self._centroid
    
    def _update_error(self):
        #TODO
        if not self._data:
            self._error = None
            return
        pass

    def get_error(self):
        error = 0
        for row_num in self._data:
            error += self._kmeans.dist(self._centroid, self._kmeans.lookup_index(row_num).values) ** 2
        return error

    
    def get_data(self):
        return self._data