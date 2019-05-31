# gradient descent-based algorithms for minimizing cost


def gradient_descend(grad_fn, init_theta, data_stream, alpha):
    """
    Simple gradient descent algorithm for minimizing objective
    """

    curr_theta = init_theta

    for aux_data in data_stream():

        curr_grad = grad_fn(curr_theta, aux_data)
        curr_theta -= alpha * curr_grad
        yield curr_theta




def mini_batch_generator():
    pass