#gradient descent-based algorithms for minimizing cost and associated helper functions

import numpy as np

def squared_error_cost(train_ex,sol,theta):
    h = train_ex @ theta
    return np.sum(np.power(h - sol,2))/train_ex.shape[0]

def squared_error_cost_gradient(train_ex, sol, theta):
    h = train_ex @ theta 
    return train_ex.T @ (h - sol)/train_ex.shape[0]

def sigmoid(train_ex, theta):
    z = -train_ex @ theta
    return 1/(1+np.exp(z))

def logistic_cost(train_ex,sol,theta):
    h = sigmoid(train_ex,theta)
    return -(sol.T @ np.log(h) + (1-sol).T @ np.log(1-h))/train_ex.shape[0]    

def logistic_cost_gradient(train_ex,sol,theta):
    return train_ex.T @ (sigmoid(train_ex,theta)-sol)/train_ex.shape[0]

def softmax(train_ex,theta):
    e = np.exp(train_ex @ theta)
    return e/(np.sum(e))

def cross_entropy_cost(train_ex,sol,theta):
    h = softmax(train_ex,theta)
    return -np.sum(sol.T @ np.log(h) + (1-sol).T @ np.log(1-h))/train_ex.shape[0]/theta.shape[1]

def cross_entropy_cost_gradient(train_ex, sol, theta):
    return train_ex.T @ (softmax(train_ex,theta) - sol)/train_ex.shape[0]/theta.shape[1]

def classification_accuracy(train_ex,sol,theta):
    guess = np.zeros(sol.shape)
    guess_idx = np.argmax(softmax(train_ex,theta), axis=1)
    guess[range(guess.shape[0]),guess_idx] = 1
    return np.sum(np.square(guess - sol))/train_ex.shape[0]/2

def column_normalize(array):
    mean_vals = array.mean(axis=0)
    std_vals = array.std(axis=0) + .000001
    return (np.subtract(array,mean_vals)/std_vals, (mean_vals,std_vals))

def gradient_descent(train_ex, solutions, grad_fn, alpha=.5, iterations=1000, track_err = False, error_fn = None,init_theta = None,track_progress = False, num_logs= 100):
    if track_err:
        err = np.zeros(num_logs)
    if init_theta is None:
        init_theta = np.zeros(train_ex.shape[1],)
    curr_theta = init_theta
    for i in range(iterations):
        if track_progress and i%(iterations/100) == 0:
            print("{}% complete".format((i/iterations)*100))
        curr_grad = grad_fn(train_ex,solutions,curr_theta)
        if track_err:
            if error_fn is None:
                print("ERROR FUNCTION MUST BE PASSED WHEN ERROR TRACKING IS ENABLED!!")
                raise ValueError 
            if i%int(iterations/num_logs) == 0:
                err[int(i*num_logs/iterations)] = error_fn(train_ex,solutions,curr_theta)
        curr_theta -= alpha*curr_grad
    if track_err:
        return (curr_theta,err) 
    return curr_theta

def sgd_optimize(train_ex, solutions, grad_fn, mini_batch_size, alpha = .5, iterations = 1000, track_err = False, error_fn = None, init_theta = None,track_progress = False,num_logs = 100):
    if track_err:
        err = np.zeros(num_logs)
    if init_theta is None:
        init_theta = np.zeros(train_ex.shape[1],)
    curr_theta = init_theta
    for i in range(iterations):
        if track_progress and i%(iterations/100) == 0:
            print("{}% complete".format((i/iterations)*100))
        indices = np.random.choice(train_ex.shape[0],mini_batch_size)
        min_batch = np.take(train_ex,indices,axis=0)
        sol_batch = np.take(solutions, indices, axis=0)
        curr_grad = grad_fn(min_batch, sol_batch, curr_theta)
        if track_err:
            if error_fn is None:
                print("ERROR FUNCTION MUST BE PASSED WHEN ERROR TRACKING IS ENABLED!!")
                raise ValueError 
            if i%int(iterations/num_logs) == 0:
                err[int(i*num_logs/iterations)] = error_fn(train_ex,solutions,curr_theta)
        curr_theta -= alpha*curr_grad 
    if track_err:
        return (curr_theta,err) 
    return curr_theta


def adam_optimize(train_ex, solutions, grad_fn, mini_batch_size, alpha = .5, gamma = .01, iterations = 1000, track_err = False,error_fn = None, init_theta = None, track_progress = False, num_logs = 100):
    if track_err:
        err = np.zeros(num_logs) 
    if init_theta is None:
        init_theta = np.zeros(train_ex.shape[1],)
    curr_theta = init_theta
    y = np.zeros(init_theta.shape)
    y_last = np.zeros(init_theta.shape)
    for i in range(iterations):
        if track_progress and i%(iterations/100) == 0:
            print("{}% complete".format((i/iterations)*100))
        indices = np.random.choice(train_ex.shape[0],mini_batch_size)
        min_batch = np.take(train_ex,indices,axis=0)
        sol_batch = np.take(solutions, indices, axis=0)
        curr_grad = grad_fn(min_batch, sol_batch, curr_theta)
        if track_err:
            if error_fn is None:
                print("ERROR FUNCTION MUST BE PASSED WHEN ERROR TRACKING IS ENABLED!!")
                raise ValueError
            if i%int(iterations/num_logs) == 0:
                err[int(i*num_logs/iterations)] = error_fn(train_ex,solutions,curr_theta)
        y = curr_theta - alpha*curr_grad
        curr_theta = (1-gamma)*y + gamma*y_last
        y_last = y
    if track_err:
        return (curr_theta,err) 
    return curr_theta
