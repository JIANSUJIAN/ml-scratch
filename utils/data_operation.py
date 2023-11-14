from __future__ import division
import numpy as np
import math
import sys


def calculate_entropy(y):
    """ Calculate the entropy of label array y """
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy


def mean_squared_error(y_true, y_pred):
    """ Returns the mean squared error between y_true and y_pred """
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse


def calculate_variance(X):
    """ Return the variance of the features in dataset X """
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
    
    return variance


def calculate_std_dev(X):
    """ Calculate the standard deviations of the features in dataset X """
    std_dev = np.sqrt(calculate_variance(X))
    return std_dev


def euclidean_distance(x1, x2):
    """ Calculates the l2 distance between two vectors """
    distance = 0
    # Squared distance between each coordinate
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the accuracy of predictions made by a classification model.

    Accuracy is defined as the proportion of correctly predicted observations to the total observations.
    This function compares each predicted label (y_pred) with the corresponding true label (y_true) and
    calculates the fraction of correct predictions.

    Parameters:
    -----------
    y_true : np.ndarray
        The true labels. An array of actual class labels.
    y_pred : np.ndarray
        The predicted labels. An array of class labels predicted by the model.

    Returns:
    --------
    float
        The accuracy of the predictions, ranging from 0 to 1.
        A value of 1 means perfect accuracy, and 0 means no correct predictions.
    """
    # np.sum(y_true == y_pred, axis=0) counts the number of times the predicted label matches the true label.
    # This is done element-wise: for each element in y_true and y_pred, it checks if they are equal.
    # If they are equal (i.e., the prediction is correct), it counts as 1, otherwise as 0.

    # The total count of correct predictions is then divided by the length of y_true (total number of observations).
    # This division gives the proportion of correct predictions, which is the accuracy.
    # The result is a float value between 0 and 1, where 1 indicates perfect accuracy.
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)

    return accuracy


def calculate_covariance_matrix(X, Y=None):
    """ Calculate the covariance matrix for the dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)
 

def calculate_correlation_matrix(X, Y=None):
    """ Calculate the correlation matrix for the dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance = (1 / n_samples) * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(Y), 1)
    correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_y.T))

    return np.array(correlation_matrix, dtype=float)