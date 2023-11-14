import numpy as np
import math
from typing import Optional, Tuple

from ..deep_learning.activation_functions import Sigmoid

class LogisticRegression:
    """
    Logistic Regression Classifier.

    This classifier uses a logistic function to model a binary dependent variable. 
    It's a type of regression analysis used for predicting binary outcomes (1/0, Yes/No, True/False).

    Attributes:
    -----------
    learning_rate : float
        The step length used for updating the parameters during training.
    gradient_descent : bool
        Whether to use gradient descent (True) or batch optimization (False) for training.
    param : np.ndarray
        Coefficients for the logistic regression model.
    sigmoid : Sigmoid
        Sigmoid function used as the activation function.
    """

    def __init__(self, learning_rate: float = 0.1, gradient_descent: bool = True) -> None:
        """
        Initialize the logistic regression model with given learning rate and optimization technique.

        Parameters:
        -----------
        learning_rate : float
            The step length used for updating the parameters during training.
        gradient_descent : bool
            Whether to use gradient descent (True) or batch optimization (False) for training.
        """

        self.param: Optional[np.ndarray] = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid

