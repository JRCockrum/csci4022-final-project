import pandas as pd

class Cluster:
    def __init__(self, kmodes):
        self._centroid = None
        self._data =  [] # list of indexes
        self._kmodes = kmodes

    def get_centroid(self):
        return self._centroid

    def get_data(self):
        return self._data
    
    def has_row(self, row_num):
        return row_num in self._data
    
    def add_row(self, row):
        if not self.has_row(row):
            self._data.append(row)

    def drop_row(self, row):
        if self.has_row(row):
            self._data.remove(row)

    def update_centroid(self):
        if not self._data:
            self._centroid = None
            return
        
        row_vals = [self._kmodes.get_row(idx) for idx in self._data]
        df = pd.DataFrame(row_vals)
        self._centroid = df.mode().iloc[0].values
