#gradient descent algorithm for minimizing cost

import autograd.numpy as np
from autograd import elementwise_grad as egrad
import itertools

def squared_error_cost(a,b):
    return np.sum(np.power(a - b,2))

def linear_model(dat, theta):
    return np.dot(dat,theta)

def gradient_descent(train_ex, solutions, model=linear_model, cost_fn=squared_error_cost, alpha=.001, iterations=10000):
    global_cost = lambda theta: cost_fn(model(train_ex,theta), solutions)
    grad_fn = egrad(global_cost)
    curr_theta = np.zeros((train_ex.shape[1],1))
    for i in range(iterations):
        curr_grad = grad_fn(curr_theta)
        curr_theta -= alpha*curr_grad
    return curr_theta

def minibatch_gradient_descent(train_ex, solutions, model=linear_model, cost_fn=squared_error_cost,alpha=.001, batch_size=1000, epochs=1000):
    if train_ex.shape[1] < batch_size:
        return gradient_descent(train_ex, solutions, model, cost_fn, alpha, 10000)

    curr_theta = np.zeros((train_ex.shape[1],1))

    num_batches = math.floor(train_ex.shape[0]/batch_size)
    batch_data = np.reshape(train_ex[0:num_batches*batch_size,:], (batch_size, train_ex.shape[1], num_batches))
    last_batch = train_ex[num_batches*batch_size:,:]
    for i in range(epochs):
        for x in itertools.chain(batch_data,[last_batch]):        
            global_cost = lambda theta: cost_fn(model(x,theta), solutions)
            grad_fn = egrad(global_cost)
            curr_grad = grad_fn(curr_theta)        
            curr_theta -= alpha*curr_grad 
     return curr_theta
