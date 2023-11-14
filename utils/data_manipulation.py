from __future__ import division
from itertools import combinations_with_replacement
from typing import Optional
from typing import Tuple

import numpy as np
import math
import sys

def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Generate polynomial features for the input data.

    Parameters:
    -----------
    X : np.ndarray
        Original features, shape (n_samples, n_features).
        
    degree : int
        The maximum degree of the polynomial.

    Returns:
    --------
    np.ndarray
        Augmented features of shape (n_samples, n_output_features), where 
        n_output_features is determined by the number of combinations of the
        original features up to the given polynomial degree.

    Example:
    --------
    If X is of shape (100, 2) and degree is 2, the resulting shape will be 
    (100, 6) because we have [1, a, b, a^2, a*b, b^2] as features.
    """
    
    n_samples, n_features = X.shape

    def get_combinations() -> list:
        """Get combinations of feature indices up to the given degree."""
        combs = [combinations_with_replacement(range(n_features), i) for i in range(1, degree + 1)]
        return [item for sublist in combs for item in sublist]

    combinations = get_combinations()
    n_output_features = len(combinations)
    X_poly = np.empty((n_samples, n_output_features))

    for i, indices in enumerate(combinations):  
        X_poly[:, i] = np.prod(X[:, indices], axis=1)

    return X_poly


def normalize(X: np.ndarray, axis: int = -1, order: int = 2) -> np.ndarray:
    """
    Normalize the dataset X.

    This function scales the input features such that each feature has a unit norm.
    This is a common preprocessing step to ensure that all features contribute equally
    to the result and improve numerical stability.

    Parameters:
    -----------
    X : np.ndarray
        The data to be normalized.
    axis : int, optional
        The axis in the data that represents the features. 
        -1 means the last axis. The default is -1.
    order : int, optional
        The order of the norm to use when normalizing. The default is 2, 
        which is the standard L2 norm.

    Returns:
    --------
    np.ndarray
        The normalized data.
    """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1  # Replace 0s with 1s to avoid division by zero
    return X / np.expand_dims(l2, axis)


def shuffle_data(X: np.ndarray, y: np.ndarray, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly shuffle the samples in X and y.

    Ensures that corresponding features and labels are kept together after shuffling.
    This is often used for mixing up the dataset before splitting into train and test sets or 
    before batch-based training in machine learning.

    Parameters:
    ----------
    X : np.ndarray
        Feature dataset to be shuffled.
    y : np.ndarray
        Corresponding label or target dataset to be shuffled.
    seed : Optional[int]
        An optional random seed for reproducibility of the shuffle.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        The shuffled feature and label/target datasets.
    """
    # Set the random seed if specified
    if seed:
        np.random.seed(seed)

    # Generate a sequence of indices and shuffle them
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    # Use the shuffled indices to reorder the datasets
    return X[idx], y[idx]



def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.5, shuffle: bool = True, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into training and testing sets.

    Parameters:
    ----------
    X : np.ndarray
        The input feature dataset.
    y : np.ndarray
        The target values corresponding to X.
    test_size : float, optional (default=0.5)
        The proportion of the dataset to include in the test split. Should be between 0.0 and 1.0.
    shuffle : bool, optional (default=True)
        Whether or not to shuffle the data before splitting. If shuffle is False, then the split is performed without shuffling.
    seed : Optional[int], optional (default=None)
        A seed to ensure reproducibility when shuffling.

    Returns:
    --------
    X_train, X_test, y_train, y_test : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The split training and testing sets.
    """

    # Shuffle the dataset if specified
    if shuffle:
        X, y = shuffle_data(X, y, seed)

    # Calculate the splitting index
    split_index = len(y) - int(len(y) * test_size)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test
