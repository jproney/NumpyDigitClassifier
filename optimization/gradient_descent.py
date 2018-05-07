#gradient descent algorithm for minimizing cost

import autograd.numpy as np
from autograd import elementwise_grad as egrad
import itertools

def squared_error_cost(a,b):
    return np.sum(np.power(a - b,2))

def linear_model(dat, theta):
    return np.dot(dat,theta)

def gradient_descent(train_ex, solutions, model=linear_model, cost_fn=squared_error_cost, alpha=.001, iterations=100000):
    global_cost = lambda theta: cost_fn(model(train_ex,theta), solutions)
    grad_fn = egrad(global_cost)
    curr_theta = np.zeros((train_ex.shape[1],1))
    for i in range(iterations):
        curr_grad = grad_fn(curr_theta)
        curr_theta -= alpha*curr_grad
    return curr_theta
