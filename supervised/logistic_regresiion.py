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
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X: np.ndarray) -> None:
        """
        Initialize model parameters to small random values close to zero.

        This helps in breaking the symmetry and ensures that the model learns different features.

        Parameters:
        -----------
        X : np.ndarray
            Training data, used to determine the number of features.
        """
        # Number of features in the dataset
        n_features = X.shape[1]

        # Setting the limit for the uniform distribution based on number of features
        # This helps in initializing the weights to small random values
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X: np.ndarray, y: np.ndarray, n_iterations: int = 4000) -> None:
        """
        Train the logistic regression model using the provided dataset.

        The model is trained either using gradient descent or batch optimization, 
        depending on the 'gradient_descent' attribute.

        Parameters:
        -----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Target binary values (e.g., 0 or 1).
        n_iterations : int
            Number of iterations for optimizing the model parameters.
        """

        self._initialize_parameters(X)

        for _ in range(n_iterations):
            # Linear combination of inputs and weights
            linear_output = X.dot(self.param)

            # Applying the sigmoid function to get the probabilities
            y_pred = self.sigmoid(linear_output)

            # Parameter update rules
            if self.gradient_descent:
                # Gradient Descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.
                # In the context of logistic regression, it is used to minimize the loss function (typically cross-entropy loss).

                # The gradient (or derivative) of the loss function with respect to the parameters gives the direction of steepest ascent.
                # By moving in the opposite direction (steepest descent), we can iteratively adjust the parameters to minimize the loss.

                # Here, (y - y_pred) is the error between the actual labels and predicted probabilities.
                # This error is then multiplied with the input features (X) to compute the gradient of the loss with respect to each parameter.
                # The product (y - y_pred).dot(X) gives a vector where each element represents the partial derivative of the loss with respect to the corresponding parameter.

                # The parameters are then updated by subtracting a fraction of this gradient.
                # The fraction is determined by the learning rate, which controls how big a step we take towards the minimum.
                # A smaller learning rate might lead to slower convergence, while a larger learning rate can overshoot the minimum.

                self.param -= self.learning_rate * -(y - y_pred).dot(X)
            else:
                # Batch Optimization, in this context, refers to a more sophisticated approach compared to simple gradient descent.
                # It involves using the second-order information of the loss function (like the Hessian matrix) for parameter updates.

                # The diag_gradient is a diagonal matrix where each diagonal element is the derivative of the sigmoid function
                # with respect to the corresponding linear output. This derivative is crucial for understanding how a small change
                # in the linear output (before applying the sigmoid function) would affect the probability (after applying the sigmoid).

                # np.linalg.pinv computes the Moore-Penrose pseudoinverse of a matrix. This is used here instead of a simple matrix inverse
                # to handle cases where the matrix may not be invertible (a common scenario in machine learning problems).
                # The pseudoinverse provides a best-fit (least squares) solution to a linear matrix equation, which is essential for our optimization.

                # The rest of the expression involves matrix operations that effectively compute the update rule for the parameters
                # by incorporating information about the curvature of the loss function. This can lead to more efficient and stable convergence,
                # especially in cases where the loss surface is complex.

                diag_gradient = np.diag(self.sigmoid.gradient(linear_output))
                self.param = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).dot(X.T).dot(diag_gradient.dot(X).dot(self.param) + y - y_pred)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class labels for the provided input data.
    
        The logistic regression model computes a linear combination of input features and weights,
        then applies a sigmoid function to these results. The sigmoid function's output is a probability
        value between 0 and 1. This method applies a threshold of 0.5 to these probabilities to determine
        the class labels. Values above 0.5 are classified as 1, and values below 0.5 are classified as 0.
    
        Parameters:
        -----------
        X : np.ndarray
            Input features for prediction. Each row represents a sample and each column represents a feature.
    
        Returns:
        --------
        np.ndarray
            Predicted binary class labels (0 or 1) for each sample in X.
        """
        # Multiply input features (X) with the model's parameters (weights) to get the linear combination.
        # This linear combination is then passed through the sigmoid function.
        # The sigmoid function maps the input values (which can be any real number) to a probability value between 0 and 1.
        linear_combination = X.dot(self.param)
        sigmoid_output = self.sigmoid(linear_combination)
    
        # Apply a threshold of 0.5 to the sigmoid output to perform binary classification.
        # np.round rounds the probabilities: values >= 0.5 are rounded up to 1, and values < 0.5 are rounded down to 0.
        # The result is an array of 0s and 1s, representing the predicted class labels.
        # astype(int) ensures that the output is an integer array.
        y_pred = np.round(sigmoid_output).astype(int)
    
        return y_pred
    


