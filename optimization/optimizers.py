# gradient descent-based algorithms for minimizing cost
import numpy as np
import math


def gradient_descend(grad_fn, init_theta, data_stream, alpha):
    """
    Simple gradient descent algorithm for minimizing objective
    """

    curr_theta = np.copy(init_theta)

    for aux_data in data_stream:
        curr_grad = grad_fn(curr_theta, aux_data)
        curr_theta -= alpha * curr_grad
        yield curr_theta


def dummy_data_stream(steps):
    for x in range(steps):
        yield None


def full_batch_stream(X, y, steps):
    for x in range(steps):
        yield X, y


def mini_batch_stream(X, y, batch_size, epochs):
    n = X.shape[0]
    ind = [x for x in range(n)]
    n_batch = math.ceil(n / batch_size)
    for i in range(epochs):
        np.random.shuffle(ind)
        for j in range(n_batch):
            batch_ind = ind[j*batch_size:min((j+1)*batch_size, n)]
            yield X[batch_ind, :], y[batch_ind]
