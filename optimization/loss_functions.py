# Vectorized loss functions and gradients. Capital letters are matrices. See optimization.ipynb for derivations.

import numpy as np


def squared_error_cost(X, y, theta):
    y_hat = X @ theta
    return np.sum(np.power(y_hat - y, 2)) / X.shape[0]


def squared_error_cost_gradient(X, y, theta):
    y_hat = X @ theta
    return X.T @ (y_hat - y) / X.shape[0]


def sigmoid(X, theta):
    z = -X @ theta
    return 1 / (1 + np.exp(z))


def softmax(X, Theta):
    h = X @ Theta
    m = np.max(h)
    e = np.exp(h - m)
    return e / np.sum(e)


def cross_entropy_cost(X, Y, Theta):
    h = softmax(X, Theta)
    return -np.sum(Y * np.log(h)) / X.shape[0]


def cross_entropy_cost_gradient(X, Y, Theta):
    return X.T @ (softmax(X, Theta) - Y) / X.shape[0]