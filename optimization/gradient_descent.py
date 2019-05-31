# gradient descent-based algorithms for minimizing cost and associated helper functions

import numpy as np


def squared_error_cost(X, y, theta):
    y_hat = X @ theta
    return np.sum(np.power(y_hat - y, 2)) / X.shape[0]


def squared_error_cost_gradient(X, y, theta):
    y_hat = X @ theta
    return X.T @ (y_hat - y) / X.shape[0]


def sigmoid(X, theta):
    z = -X @ theta
    return 1 / (1 + np.exp(z))


def logistic_cost(train_ex, sol, theta):
    h = sigmoid(train_ex, theta)
    return -(sol.T @ np.log(h) + (1 - sol).T @ np.log(1 - h)) / train_ex.shape[0]


def logistic_cost_gradient(train_ex, sol, theta):
    return train_ex.T @ (sigmoid(train_ex, theta) - sol) / train_ex.shape[0]


def softmax(train_ex, theta):
    h = train_ex @ theta
    m = np.max(h)
    e = np.exp(h - m)
    return e / (np.sum(e))


def cross_entropy_cost(train_ex, sol, theta):
    h = softmax(train_ex, theta)
    return -np.sum(sol * np.log(h)) / train_ex.shape[0] / theta.shape[1]


def cross_entropy_cost_gradient(train_ex, sol, theta):
    return train_ex.T @ (softmax(train_ex, theta) - sol) / train_ex.shape[0] / theta.shape[1]


def classification_accuracy(train_ex, sol, theta):
    guess = np.zeros(sol.shape)
    guess_idx = np.argmax(softmax(train_ex, theta), axis=1)
    guess[range(guess.shape[0]), guess_idx] = 1
    return np.sum(np.square(guess - sol)) / train_ex.shape[0] / 2


def column_normalize(array):
    mean_vals = array.mean(axis=0)
    std_vals = array.std(axis=0) + .00001
    return (np.subtract(array, mean_vals) / std_vals, (mean_vals, std_vals))


def gradient_descent(train_ex, solutions, grad_fn, alpha=.5, epochs=100, track_err=False, error_fn=None,
                     init_theta=None, track_progress=False, num_logs=100):
    if track_err:
        err = np.zeros(num_logs)
    if init_theta is None:
        init_theta = np.zeros((train_ex.shape[1], 1))
    curr_theta = init_theta
    for i in range(epochs):
        if track_progress and i % (epochs / 100) == 0:
            print("{}% complete".format((i / epochs) * 100))
        curr_grad = grad_fn(train_ex, solutions, curr_theta)
        if track_err:
            if error_fn is None:
                print("ERROR FUNCTION MUST BE PASSED WHEN ERROR TRACKING IS ENABLED!!")
                raise ValueError
            if i % int(epochs / num_logs) == 0:
                err[int(i * num_logs / epochs)] = error_fn(train_ex, solutions, curr_theta)
        curr_theta -= alpha * curr_grad
    if track_err:
        return (curr_theta, err)
    return curr_theta


def sgd_optimize(train_ex, solutions, grad_fn, mini_batch_size, alpha=.5, epochs=100, track_err=False, error_fn=None,
                 init_theta=None, track_progress=False, num_logs=100):
    batches_per_epoch = int(train_ex.shape[0] / mini_batch_size)
    total_iters = batches_per_epoch * epochs
    if track_err:
        num_logs = min(num_logs, total_iters)
        err = np.zeros(num_logs)
    if init_theta is None:
        init_theta = np.zeros((train_ex.shape[1], 1))
    curr_theta = init_theta
    for i in range(epochs):
        shuff = np.hstack((train_ex, solutions))
        np.random.shuffle(shuff)
        min_batches = shuff[:, 0:train_ex.shape[1]]
        sol_batches = shuff[:, train_ex.shape[1]:]
        for j in range(batches_per_epoch):
            curr_batch = min_batches[j * mini_batch_size:(j + 1) * mini_batch_size, :]
            curr_sol = sol_batches[j * mini_batch_size:(j + 1) * mini_batch_size, :]
            curr_grad = grad_fn(curr_batch, curr_sol, curr_theta)
            curr_theta -= alpha * curr_grad
            idx = i * batches_per_epoch + j
            if track_progress and idx % (total_iters / 100) == 0:
                print("{}% complete".format((idx / total_iters) * 100))
            if track_err:
                if error_fn is None:
                    print("ERROR FUNCTION MUST BE PASSED WHEN ERROR TRACKING IS ENABLED!!")
                    raise ValueError
                if idx % int(total_iters / num_logs) == 0:
                    err[int(idx * num_logs / total_iters)] = error_fn(train_ex, solutions, curr_theta)
    if track_err:
        return (curr_theta, err)
    return curr_theta


def adam_optimize(train_ex, solutions, grad_fn, mini_batch_size, alpha=.5, gamma=.99, lamb=0, w=200, epochs=100,
                  track_err=False, error_fn=None, init_theta=None, track_progress=False, num_logs=100):
    if track_err:
        err = np.zeros(num_logs)
    if init_theta is None:
        init_theta = np.zeros((train_ex.shape[1], 1))
    curr_theta = init_theta
    batches_per_epoch = int(train_ex.shape[0] / mini_batch_size)
    total_iters = batches_per_epoch * epochs
    grad_history = np.zeros((curr_theta.size, epochs))
    G = np.zeros(curr_theta.shape)
    moment = np.zeros(init_theta.shape)
    eps = np.ones(G.shape) * 1e-8
    for i in range(epochs):
        shuff = np.hstack((train_ex, solutions))
        np.random.shuffle(shuff)
        min_batches = shuff[:, 0:train_ex.shape[1]]
        sol_batches = shuff[:, train_ex.shape[1]:]
        for j in range(batches_per_epoch):
            curr_batch = min_batches[j * mini_batch_size:(j + 1) * mini_batch_size, :]
            curr_sol = sol_batches[j * mini_batch_size:(j + 1) * mini_batch_size, :]
            curr_grad = grad_fn(curr_batch, curr_sol,
                                curr_theta - gamma * moment)  # add momentum before computing grad
            curr_grad += lamb * curr_theta  # regularize
            grad_history[:, i] = np.ndarray.flatten(np.square(curr_grad))
            G = np.sum(grad_history[:, max(0, i - w):i + 1], axis=1).reshape(curr_grad.shape)
            lrates = alpha / np.sqrt(G + eps)
            adaptive_grad = lrates * curr_grad
            moment = gamma * moment + adaptive_grad
            curr_theta -= moment
            idx = i * batches_per_epoch + j
            if track_progress and idx % (total_iters / 100) == 0:
                print("{}% complete".format((idx / total_iters) * 100))
            if track_err:
                if error_fn is None:
                    print("ERROR FUNCTION MUST BE PASSED WHEN ERROR TRACKING IS ENABLED!!")
                    raise ValueError
                if idx % int(total_iters / num_logs) == 0:
                    err[int(idx * num_logs / total_iters)] = error_fn(train_ex, solutions, curr_theta)
    if track_err:
        return (curr_theta, err)
    return curr_theta
