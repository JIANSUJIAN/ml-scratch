import numpy as np


class Sigmoid:
    """
    Represents the sigmoid activation function.

    The sigmoid function, defined as sigmoid(x) = 1 / (1 + exp(-x)), is widely used in
    machine learning, particularly in logistic regression and neural networks, for tasks
    such as binary classification. It maps input values to output values between 0 and 1.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the sigmoid of the input.

        Parameters:
        ----------
        x : np.ndarray
            The input array for which to compute the sigmoid function.

        Returns:
        -------
        np.ndarray
            The sigmoid of the input array.
        """
        return 1 / (1 + np.exp(-x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient (derivative) of the sigmoid function.

        The gradient is used in optimization algorithms, particularly during
        the backpropagation step in neural networks.

        Parameters:
        ----------
        x : np.ndarray
            The input array for which to compute the gradient.

        Returns:
        -------
        np.ndarray
            The gradient of the sigmoid at the input array.
        """
        sigmoid_x = self.__call__(x)
        return sigmoid_x * (1 - sigmoid_x)
