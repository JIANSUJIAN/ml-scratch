import numpy as np
import math
import sys
from typing import Any
from ..utils.data_manipulation import normalize, polynomial_features




class Regression(object):
    """
    Base regression model.
    
    This class models the relationship between a scalar dependent variable y and the independent variables X.
    
    Parameters
    ----------
    n_iterations : int
        The number of training iterations the algorithm will tune the weights for.
    learning_rate : float
        The step length that will be used when updating the weights.
    """
    
    def __init__(self, n_iterations: int, learning_rate: float) -> None:
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features: int) -> None:
        """
        Initialize the weights for the features using a uniform distribution.
        
        The weights are initialized in the range [-limit, limit] where limit is 1/sqrt(n_features).
        
        Parameters
        ----------
        n_features : int
            Number of features (including the bias).
        """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the regression model using gradient descent.
        
        Parameters
        ----------
        X : np.ndarray
            Training dataset.
        y : np.ndarray
            Target values.
        """
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []  # Keep track of training errors (MSE) for each iteration
        self.initialize_weights(n_features=X.shape[1])

        # Gradient descent optimization
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)
            # Calculate L2 loss (MSE + Regularization term)
            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))
            self.training_errors.append(mse)
            # Gradient of L2 loss with respect to weights
            grad_w = -(y - y_pred).dot(X) + self.regularization(self.w)
            # Update the weights
            self.w -= self.learning_rate * grad_w

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for given input dataset.
        
        Parameters
        ----------
        X : np.ndarray
            Input dataset.
            
        Returns
        -------
        np.ndarray
            Predicted values.
        """
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred

    def regularization(self, w: np.ndarray) -> np.ndarray:
        """
        Regularization term. 
        Should be overridden by subclasses if regularization is to be applied.
        
        Parameters
        ----------
        w : np.ndarray
            Current weight values.
            
        Returns
        -------
        np.ndarray
            Regularization term for the given weights. Defaults to zeros.
        """
        return np.zeros_like(w)

class LinearRegression(Regression):
    """
    Linear Regression model, a subclass of the Regression class.
    
    The Regression class serves as a base for regression models, which captures the relationship 
    between a scalar dependent variable 'y' and independent variables 'X'. This relationship is 
    typically captured through a set of weights which are adjusted during the training process.
    
    Parameters
    ----------
    n_iterations : int, optional (default=100)
        The number of training iterations the algorithm will tune the weights for if using gradient descent.
    learning_rate : float, optional (default=0.001)
        The step length that will be used when updating the weights, if using gradient descent.
    gradient_descent : bool, optional (default=True)
        If True, use gradient descent for optimization.
        If False, use batch optimization by least squares.
    """
    def __init__(self, n_iterations: int = 100, learning_rate: float = 0.001, gradient_descent: bool = True):
        self.gradient_descent = gradient_descent

        # Set no regularization for basic linear regression

        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0

        super().__init__(n_iterations=n_iterations, learning_rate=learning_rate)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the regression model to the trainning data.

        Args:
            X (np.ndarray): Training dataset.
            y (np.ndarray): Target values.
        """
        if not self.gradient_descent:
            # If not using gradient descent, compute weights using the least sqaures method
            X = np.insert(X, 0, 1, axis=1)

            # Singular Value Decomposition (SVD) of the matrix X^T X.
            # U: left singular vectors
            # S: diagonal matrix with singular values (returned as a 1D array)
            # V: right singular vectors
            U, S, V = np.linalg.svd(X.T.dot(X))

            # Convert the singular value (1D array) to a diagonal matrix.
            S = np.diag(S)

            # Compute the Moore-Penrose pseudoinverse of X^T X using the matrices from SVD.
            # This pseudoinverse provides a stable and efficient way to compute the regression weights.
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)

            # Calculate the regression weights based on the formula w = (X^T X)^+ X^T y
            # where (X^T X)^+ is the pseudoinverse of X^T X.
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            # If using gradient descent, call the fit method from the parent (Regression) class
            super().fit(X, y)


    
class l1_regularization():
    """
    L1 Regularization (Lasso Regression).
    
    Regularization term:
    R1(w) = alpha * sum(|w_i|)
    
    This encourages sparsity by driving some weights to exactly zero.
    
    Attributes:
    -----------
    Alpha: float
        Regularization strength coefficient.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
    
    def __call__(self, w: np.ndarray) -> float:
        """Compute the L1 regularization term."""
        return self.alpha * np.linalg.norm(w, ord=1)

    def grad(self, w: np.ndarray) -> np.ndarray:
        """Gradient of the L1 regularization term. Returns the sign of the weights."""
        return self.alpha * np.sign(w)

class l2_regularization():
    """ 
    L2 Regularization (Ridge Regression).
    
    Regularization term:
    R2(w) = 0.5 * alpha * sum(w_i^2)
    
    This pushes weights towards zero but doesn't set them exactly to zero.
    
    Attributes:
    -----------
    alpha: float
        Regularization strength coefficient.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
    
    def __call__(self, w: np.ndarray) -> float:
        """"Compute the L2 regularization term."""
        return self.alpha * 0.5 * w.T.dot(w)
    
    def grad(self, w: np.ndarray) -> np.ndarray:
        """Gradient of the L2 regularization term. Directly proportional to the weights."""
        return self.alpha * w

class l1_l2_regularization():
    """ 
    Elastic Net Regularization.
    
    Regularization term:
    RElasticNet(w) = alpha * (l1_ratio * sum(|w_i|) + 0.5 * (1 - l1_ratio) * sum(w_i^2))
    
    Combines L1 and L2 regularization methods.
    
    Attributes:
    -----------
    alpha: float
        Regularization strength coefficient.
    l1_ratio: float
        Ratio of L1 regularization. (1 - l1_ratio) gives the L2 regularization ratio.
    """

    def __init__(self, alpha: float, l1_ratio: float = 0.5) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
    
    def __call__(self, w: np.ndarray) -> float:
        """Compute the Elastic Net regularization term."""
        l1_contr = self.l1_ratio * np.linalg.norm(w, ord=1)
        l2_contr = (1 - self.l1_ratio) * 0.5 * w.T.dot(w)
        return self.alpha * (l1_contr + l2_contr)
    
    def grad(self, w: np.ndarray) -> np.ndarray:
        """Gradient of the Elastic Net regularization term. Combination of L1 and L2 components."""
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr)

class LassoRegression(Regression):
    """
    Lasso Regression Model.

    This is a form of linear regression that includes a regularization term. The
    regularization term discourages overly complex models which can lead to overfitting.
    The strength of the regularization is controlled by the `reg_factor`. Lasso Regression
    specifically uses L1 regularization which can lead to sparse models with some feature
    coefficients becoming exactly zero.

    Parameters:
    -----------
    degree : int
        Degree of the polynomial for feature transformation.
        
    reg_factor : float
        Regularization factor. Controls the strength of regularization. Higher values
        mean stronger regularization. Regularization will lead to feature shrinkage,
        potentially driving some coefficients to zero.
        
    n_iterations : int, optional (default=3000)
        Number of training iterations the algorithm will tune the weights for.
        
    learning_rate : float, optional (default=0.01)
        Step length used when updating weights during training.

    Attributes:
    -----------
    degree : int
        Degree of polynomial used for feature transformation.
        
    regularization : instance of Regularization
        Regularization used for this model, in this case L1 (Lasso).
    """

    def __init__(self, degree: int, reg_factor: float,
                 n_iterations: int = 3000,
                 learning_rate: float = 0.01) -> None:
        self.degree = degree
        self.regularization = l1_regularization(alpha=reg_factor)
        super().__init__(n_iterations=n_iterations, learning_rate=learning_rate)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Lasso regression model.

        Transforms input features to polynomial features based on specified degree
        and then normalizes them. It then uses the transformed and normalized features
        to train the model.
        """
        X = normalize(polynomial_features(X, degree=self.degree))
        super().fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target using the Lasso regression model.

        Transforms input features to polynomial features based on specified degree
        and then normalizes them. It then uses the transformed and normalized features
        for prediction.
        """
        X = normalize(polynomial_features(X, degree=self.degree))
        return super().predict(X) 


class PolynomialRegression(Regression):
    """
    Extends the Regression class to perform polynomial regression.
    
    This class takes a simple linear approach and extends it to fit non-linear relationships
    by transforming the original features into polynomial features of a specified degree.

    Attributes:
    ----------
    degree : int
        The degree of the polynomial that the independent variable X will be transformed into.
        For example, if degree=2, features will be transformed into their square.

    n_iterations : float
        The number of training iterations the algorithm will perform to tune the weights.

    learning_rate : float
        The step length that will be taken when updating the weights during training.
    """
    def __init__(self, degree: int, n_iterations: float = 3000, learning_rate: float = 0.001) -> None:
        self.degree = degree
        # No regularization is applied in polynomial regression by default
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super().__init__(n_iterations=n_iterations,
                                                   learning_rate=learning_rate)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the polynomial regression model on the training data.

        The method transforms the original features into polynomial features of the specified degree
        and then fits a linear regression model on these transformed features.

        Parameters:
        ----------
        X : np.ndarray
            The training data's features.
        
        y : np.ndarray
            The target values.
        """
        X_transformed = polynomial_features(X, degree=self.degree)
        super().fit(X_transformed, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted polynomial regression model.

        The method transforms the original features into polynomial features of the specified degree
        and then uses the fitted model to make predictions.

        Parameters:
        ----------
        X : np.ndarray
            The data for which to make predictions.
        
        Returns:
        -------
        np.ndarray
            The predicted values.
        """
        X_transformed = polynomial_features(X, degree=self.degree)
        return super(PolynomialRegression, self).predict(X_transformed)

    
class RidgeRegression(Regression):
    """
    Implements Ridge Regression, a variant of Linear Regression with L2 regularization.

    Ridge Regression, also known as Tikhonov regularization, adds a regularization term
    to the cost function, which helps in reducing model complexity and preventing overfitting.
    The strength of the regularization is controlled by a parameter.

    Attributes:
    ----------
    reg_factor : float
        The regularization factor, denoted as lambda in the regularization term.
        Higher values indicate stronger regularization.

    n_iterations : float
        The number of training iterations the algorithm will perform to adjust the weights.

    learning_rate : float
        The step length that will be taken when updating the weights during training.
    """
    def __init__(self, reg_factor: float, n_iterations: float = 1000, learning_rate: float = 0.001) -> None:
        self.regularization = l2_regularization(alpha=reg_factor)
        super().__init__(n_iterations=n_iterations,
                                              learning_rate=learning_rate)

    
class PolynomialRidgeRegression(Regression):
    """
    Polynomial Ridge Regression is an extension of Ridge Regression that allows for
    polynomial relationships between the features and the target variable.
    
    It first transforms the input data into polynomial features and then applies
    Ridge Regression with L2 regularization to the transformed data. This approach
    is beneficial for modeling more complex relationships while still controlling
    overfitting through regularization.
    
    Attributes:
    ----------
    degree : int
        The degree of the polynomial transformation applied to the independent variable X.
    
    reg_factor : float
        The regularization factor (lambda) used for L2 regularization.
    
    n_iterations : float
        The number of training iterations for adjusting the model's weights.
    
    learning_rate : float
        The step size used for updating the weights during training.
    
    gradient_descent : bool
        Indicates whether to use gradient descent (True) or another optimization method (False).
    """
    def __init__(self, degree: int, reg_factor: float, n_iterations: float = 3000, learning_rate: float = 0.01, gradient_descent: bool = True) -> None:
        self.degree = degree
        self.regularization = l2_regularization(alpha=reg_factor)
        self.gradient_descent = gradient_descent
        super().__init__(n_iterations=n_iterations,
                                                        learning_rate=learning_rate,
                                                        gradient_descent=gradient_descent)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the Polynomial Ridge Regression model to the training data.
        
        Parameters:
        X : np.ndarray
            The training input samples.
        y : np.ndarray
            The target values.
        """
        X_transformed = normalize(polynomial_features(X, degree=self.degree))
        super().fit(X_transformed, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for input samples using the trained Polynomial Ridge Regression model.
        
        Parameters:
        X : np.ndarray
            The input samples.
        
        Returns:
        np.ndarray
            The predicted target values.
        """
        X_transformed = normalize(polynomial_features(X, degree=self.degree))
        return super().predict(X_transformed)


