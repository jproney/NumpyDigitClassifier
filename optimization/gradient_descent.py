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

def gradient_descent(train_ex, solutions, grad_fn, alpha=.0001, iterations=1000, normalize = True, track_err = False):
    if track_err:
        err = np.zeros(iterations)
    if normalize:
        (train_ex, train_norm_params) = column_normalize(train_ex)
        (solutions,sol_norm_params) = column_normalize(solutions)
    curr_theta = np.zeros((train_ex.shape[1],))
    for i in range(iterations):
        curr_grad = grad_fn(train_ex,solutions,curr_theta)
        if track_err:
            err[i] = squared_error_cost(train_ex @ curr_theta,solutions)
        curr_theta -= alpha*np.ndarray.flatten(curr_grad)
    if normalize:
        if track_err:
            return (curr_theta,train_norm_params,sol_norm_params,err) 
        return (curr_theta,train_norm_params,sol_norm_params)
    return curr_theta

def sgd_optimize(train_ex, solutions, grad_fn, mini_batch_size, alpha = .0001, iterations = 1000, normalize = True, track_err = False):
    if track_err:
        err = np.zeros(iterations)
    if normalize: 
        (train_ex, train_norm_params) = column_normalize(train_ex)
        (solutions,sol_norm_params) = column_normalize(solutions)
    curr_theta = np.zeros((train_ex.shape[1],))
    for i in range(iterations):
        indices = np.random.choice(train_ex.shape[0],mini_batch_size)
        min_batch = np.take(train_ex,indices,axis=0)
        sol_batch = np.take(solutions, indices, axis=0)
        curr_grad = grad_fn(min_batch, sol_batch, curr_theta)
        if track_err:
            err[i] = squared_error_cost(min_batch @ curr_theta,sol_batch)
        curr_theta -= alpha*np.ndarray.flatten(curr_grad)
    if normalize: 
        if track_err:
            return (curr_theta,train_norm_params,sol_norm_params,err) 
        return (curr_theta,train_norm_params,sol_norm_params)
    return curr_theta


def adam_optimize(train_ex, solutions, grad_fn, mini_batch_size, alpha = .0001, gamma = .99, eta = .00005, iterations = 1000, normalize = True, track_err = False):
    if track_err:
        err = np.zeros(iterations)
    if normalize: 
        (train_ex, train_norm_params) = column_normalize(train_ex)
        (solutions,sol_norm_params) = column_normalize(solutions)
    curr_theta = np.zeros((train_ex.shape[1],))
    momentum = np.zeros((train_ex.shape[1],))
    for i in range(iterations):
        indices = np.random.choice(train_ex.shape[0],mini_batch_size)
        min_batch = np.take(train_ex,indices,axis=0)
        sol_batch = np.take(solutions, indices, axis=0)
        curr_grad = grad_fn(min_batch, sol_batch, curr_theta)
        if track_err:
            err[i] = squared_error_cost(min_batch @ curr_theta,sol_batch)
        momentum = gamma*momentum + eta*np.ndarray.flatten(curr_grad)
        curr_theta -= (alpha*np.ndarray.flatten(curr_grad) + momentum)
    if normalize: 
        if track_err:
            return (curr_theta,train_norm_params,sol_norm_params,err) 
        return (curr_theta,train_norm_params,sol_norm_params)
    return curr_theta
