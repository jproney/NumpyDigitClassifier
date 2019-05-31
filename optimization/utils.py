# Helper functions

import numpy as np
from optimization import loss_functions as lf


def softmax_classification_accuracy(train_ex,sol,theta):
    guess = np.zeros(sol.shape)
    guess_idx = np.argmax(lf.softmax(train_ex,theta), axis=1)
    guess[range(guess.shape[0]),guess_idx] = 1
    return np.sum(np.square(guess - sol))/train_ex.shape[0]/2


def column_normalize(array):
    mean_vals = array.mean(axis=0)
    std_vals = array.std(axis=0) + .00001
    return np.subtract(array,mean_vals)/std_vals, (mean_vals,std_vals)