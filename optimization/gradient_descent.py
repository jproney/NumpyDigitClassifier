#gradient descent algorithm for minimizing cost

import numpy as np

def squared_error_cost(a,b):
    return np.sum(np.power(a - b,2))

def linear_squared_error_gradient(train_ex, sol, theta): 
    return np.array([[2*np.dot(np.ndarray.flatten(np.matmul(train_ex,theta)) - np.ndarray.flatten(sol), x)] for x in train_ex.T])

def column_normalize(array):
    max_vals = array.max(axis = 0)
    return (array/max_vals, max_vals)

def gradient_descent(train_ex, solutions, grad_fn, alpha=.001, iterations=100000, normalize = True):
    if normalize:
        (train_ex, col_scales) = column_normalize(train_ex)
    curr_theta = np.zeros((train_ex.shape[1],1))
    for i in range(iterations):
        curr_grad = grad_fn(train_ex,solutions,curr_theta)
        curr_theta -= alpha*curr_grad
    if normalize:
        return curr_theta
    return curr_theta
