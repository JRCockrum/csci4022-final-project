import numpy as np
import pandas as pd

class Cluster:
    def __init__(self, kmeans):
        # TODO: I'm not sure about some of these default values
        self._centroid = None
        self._data =  [] # list of indexes
        self._error = None # TODO might class this somthing else (e.g squared dist)
        self._kmeans = kmeans

    def add_row(self, row):
        self._data.append(row)
        self._update_centroid()
        # self._update_error()

    def drop_row(self, row): # row: expected to be a 1D NumPy array
        try:
            self._data.remove(row)
            if self._data:
                self._update_centroid()
                self._update_error()
            else: # if cluster is empty
                self._centroid = None
                self._error = None
        except ValueError:
            print("Error: unable to remove row because it was not found in the cluster")
            print(f"Row: {row}")

    def _update_centroid(self):
        # if not self._data:
        #     self._centroid = None
        #     return
        print("here")
        # Get the actual rows from the parent KMeans object
        rows = [self._kmeans.lookup_index(i).values for i in self._data]
        self._centroid = np.mean(rows, axis=0)

    def get_centroid(self):
        return self._centroid
    
    def _update_error(self):
        #TODO
        pass

    def get_error(self):
        return self._error
    
    def get_data(self):
        return self._data