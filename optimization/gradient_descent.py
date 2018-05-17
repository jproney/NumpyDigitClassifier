#gradient descent algorithm for minimizing cost

import numpy as np

def squared_error_cost(a,b):
    return np.sum(np.power(a - b,2))

def linear_squared_error_gradient(train_ex, sol, theta):
    h = train_ex @ theta 
    return train_ex.T @ (np.ndarray.flatten(h) - np.ndarray.flatten(sol))

def column_normalize(array):
    mean_vals = array.mean(axis=0)
    std_vals = array.std(axis=0) + .00001
    return (np.subtract(array,mean_vals)/std_vals, (mean_vals,std_vals))

def gradient_descent(train_ex, solutions, grad_fn, alpha=.001, iterations=100000, normalize = True):
    if normalize:
        (train_ex, train_norm_params) = column_normalize(train_ex)
        (solutions,sol_norm_params) = column_normalize(solutions)
    curr_theta = np.zeros((train_ex.shape[1],))
    for i in range(iterations):
        curr_grad = grad_fn(train_ex,solutions,curr_theta)
        curr_theta -= alpha*np.ndarray.flatten(curr_grad)
    if(normalize):
        return (curr_theta,train_norm_params,sol_norm_params)
    return curr_theta
