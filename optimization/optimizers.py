# gradient descent-based algorithms for minimizing cost and associated helper functions

import numpy as np


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
