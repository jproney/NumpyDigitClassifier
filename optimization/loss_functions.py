# Vectorized loss functions and gradients. Capital letters are matrices. See optimization.ipynb for derivations.

import numpy as np


def squared_error_cost(X, y, theta):
    y_hat = X @ theta
    return np.sum(np.power(y_hat - y, 2)) / X.shape[0]


def squared_error_cost_gradient(X, y, theta):
    y_hat = X @ theta
    return X.T @ (y_hat - y) / X.shape[0]


def sigmoid(z):
    return 1 / (1 + np.exp(z))


def softmax(H):
    m = np.max(H)
    e = np.exp(H - m)
    return e / np.sum(e, axis=1)[:, None]


def cross_entropy_cost(X, Y):
    H = softmax(X)
    return -np.sum(Y * np.log(H)) / H.shape[0]


def cross_entropy_cost_theta(X, Y, Theta):
    h = softmax(X @ Theta)
    return -np.sum(Y * np.log(h)) / X.shape[0]


def cross_entropy_cost_gradient(H, Y):
    return softmax(H) - Y  # N x K matrix of gradients w.r.t final network outputs pre-softmaxing


def cross_entropy_cost_gradient_theta(X, Y, Theta):
    return X.T @ (softmax(X @ Theta) - Y) / X.shape[0]
