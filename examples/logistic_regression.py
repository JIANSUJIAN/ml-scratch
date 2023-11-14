from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

from typing import Any
from ..utils.data_manipulation import normalize, train_test_split
from ..utils.data_operation import accuracy_score
from ..deep_learning.activation_functions import Sigmoid
from ..utils.misc import Plot
from ..supervised.logistic_regresiion import LogisticRegression 

def main():
    # Load the Iris dataset from scikit-learn.
    # This dataset is often used for testing out machine learning algorithms.
    data = datasets.load_iris()
    
    # Preprocess the dataset:
    # Normalize the features and filter out one class of the dataset.
    # The Iris dataset has three classes; here, we are only using two for binary classification.
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    
    # Convert the classes into a binary format for logistic regression.
    # The original classes labeled '1' are set to '0' and those labeled '2' are set to '1'.
    y[y == 1] = 0
    y[y == 2] = 1

    # Split the dataset into training and testing sets.
    # The test_size parameter indicates the proportion of the dataset to include in the test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

    # Initialize and train the Logistic Regression model.
    # Gradient descent is used for optimization.
    clf = LogisticRegression(gradient_descent=True)
    clf.fit(X_train, y_train)

    # Predict the labels of the test set.
    y_pred = clf.predict(X_test)

    # Calculate and print the accuracy of the model on the test set.
    # Accuracy is the proportion of correctly predicted observations.
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Visualize the results:
    # The test data and their predicted labels are plotted in 2D using PCA.
    # This helps in visualizing the classification boundary and how well the model has performed.
    Plot().plot_in_2d(X_test, y_pred, title="Logistic Regression", accuracy=accuracy)

if __name__ == "__main__":
    main()