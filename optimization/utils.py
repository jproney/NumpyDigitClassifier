# Helper functions

import numpy as np
import loss_functions as lf


def softmax_classification_accuracy(X, Y, Theta):
    guess = np.zeros(Y.shape)
    guess_idx = np.argmax(lf.softmax(X @ Theta), axis=1)
    guess[range(guess.shape[0]), guess_idx] = 1
    return np.sum(np.square(guess - Y))/X.shape[0]/2


def column_normalize(array):
    mean_vals = array.mean(axis=0)
    std_vals = array.std(axis=0)
    keep = std_vals > 0
    return np.subtract(array[:, keep], mean_vals[keep])/std_vals[keep], mean_vals, std_vals, keep


def to_one_hot(y, num_cats):
    n = len(y)
    one_hot_mat = np.zeros((n, num_cats))
    for ind in zip(range(n), y):
        one_hot_mat[ind] = 1
    return one_hot_mat
