import numpy as np
from ..utils.data_operation import euclidean_distance 

class KNN():
    """
    K Nearest Neighbors classifier.

    Parameters:
    -----------
    k : int
        The number of closest neighbors that will determine the class of the 
        sample that we wish to predict.
    """

    def __init__(self, k=5):
        self.k = k

    def _vote(self, neighbor_labels):
    """
    Return the most common class among the neighbor samples.

    Parameters:
    neighbor_labels : array-like
        Array of labels for the nearest neighbors found.

    Returns:
    int
        The most common class label among the neighbors.
"""
