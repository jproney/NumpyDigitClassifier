#gradient descent algorithm for minimizing cost

import autograd.numpy as np
from autograd import elementwise_grad as egrad

def squared_error_cost(a,b):
    return np.sum(np.power(a - b,2))

def gradient_descent(train_ex, solutions, cost_fn=squared_error_cost, alpha=.001, grad_thresh=.001):
    global_cost = lambda theta: cost_fn(np.dot(train_ex,theta), solutions)
    grad_fn = egrad(global_cost)
    curr_theta = np.zeros((train_ex.shape[1],1))
    while True:
        curr_grad = grad_fn(curr_theta)
        curr_theta -= alpha*curr_grad
        if(np.linalg.norm(curr_grad) < grad_thresh):
            break;
    return curr_theta


