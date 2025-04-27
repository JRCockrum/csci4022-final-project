import numpy as np
import pandas as pd
# from kmeans import Kmeans

class Cluster:
    def __init__(self, kmeans):
        self._numeric_centroid = None
        self._cat_centroid = None
        self._data =  [] # list of indexes
        self._kmeans = kmeans

    def add_row(self, row):
        if not self.has_row(row):
            self._data.append(row)
            self._update_centroids()

    def drop_row(self, row):
        try:
            self._data.remove(row)
            self._update_centroids()
        except ValueError:
            print("Error: unable to remove row because it was not found in the cluster")
            print(f"Row: {row}")

    def _update_centroids(self):
        if not self._data:
            self._numeric_centroid = None
            self._cat_centroid = None
            return
        
        # Update Numeric centroid
        numeric_rows = [self._kmeans.lookup_numeric(i) for i in self._data]
        self._numeric_centroid = np.mean(numeric_rows, axis=0)

        # Update categorical centroid
        cat_rows = [self._kmeans.lookup_cat(i) for i in self._data]
        self._cat_centroid = set().union(*cat_rows)

    def get_num_centroid(self):
        return self._numeric_centroid
    
    def get_cat_centroid(self):
        return self._cat_centroid
    
    def get_error(self):
        error = 0
        for row_num in self._data:
            error += self._kmeans.dist(self._numeric_centroid, self._kmeans.lookup_numeric(row_num), self._cat_centroid, self._kmeans.lookup_cat(row_num)) ** 2
        return error

    def get_data(self):
        return self._data
    
    def has_row(self, row_num):
        return row_num in self._data
        