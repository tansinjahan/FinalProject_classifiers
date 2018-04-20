import numpy as np
import os

def getSelection(numOfColumns, selections):
    columns = np.empty(shape=numOfColumns, dtype=bool)
    columns.fill(False)
    for col in selections:
        columns[col-1] = True
    return columns

def getSelectedColumns(npArray, selections):
    return npArray[:, getSelection(npArray.shape[1], selections)]

def currentFilePath(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    print(here)
    return os.path.join(here, filename)