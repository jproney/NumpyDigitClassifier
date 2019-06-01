import unittest
from optimization import optimizers as opt
from optimization import loss_functions as lf
import numpy as np


def fuzzy_equals(a, b, eps):
    return sum(abs(a-b)) < eps


class OptimizationTests(unittest.TestCase):

    def test_gd_parabaloid(self):

        iters = 200

        # Objective function is parabaloid: (3 - x)**2 + (7 - y)**2
        minim = np.array([3.0, 7.0])  # Global minimum exists at (3,7)

        def grad(theta, aux_data): return np.array([-2*(3-theta[0]), -2*(7-theta[1])])

        start = np.array([-4.0, 2.0])

        theta = None
        for theta in opt.gradient_descend(grad_fn=grad, data_stream=opt.dummy_data_stream(iters), alpha=.05, init_theta=start):
            pass

        assert(fuzzy_equals(theta, minim, .01))

    def test_gd_himmelblau(self):

        iters = 200

        # Himmelblau function : (x**2 + y - 11)**2 + (x + y**2 - 7)**2
        # 4 minima at: (3.0,2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
        minim1 = np.array([3.0, 2.0])
        minim2 = np.array([-2.805118, 3.131312])
        minim3 = np.array([-3.779310, -3.283186])
        minim4 = np.array([3.584428, -1.848126])

        def grad(theta, aux_data): return np.array([4*theta[0]*(theta[0]**2 + theta[1] - 11) + 2*(theta[0] + theta[1]**2 - 7),
                                                    2*(theta[0]**2 + theta[1] - 11) + 4*theta[1]*(theta[0] + theta[1]**2 - 7)])

        start = np.array([2.0, -3.0])

        theta = None
        for theta in opt.gradient_descend(grad_fn=grad, data_stream=opt.dummy_data_stream(iters), alpha=.005, init_theta=start):
            pass

        assert(fuzzy_equals(theta, minim1, .01) or fuzzy_equals(theta, minim2, .01) or fuzzy_equals(theta, minim3, .01)
               or fuzzy_equals(theta, minim4, .01))

    def test_gd_regression(self):

        iters = 10000

        theta_real = np.array([4.0, -5.3, 2.2, 11.8, 6.4, 0.8])
        X = np.random.rand(500, 6)
        y = X @ theta_real

        def grad(theta, aux_data): return lf.squared_error_cost_gradient(aux_data[0], aux_data[1], theta)

        start = np.zeros(6)

        theta = None
        for theta in opt.gradient_descend(grad_fn=grad, data_stream=opt.full_batch_stream(X, y, iters), alpha=.01, init_theta=start):
            pass

        assert(fuzzy_equals(theta, theta_real, .1))

    def test_gd_regression_minibatch(self):

        iters = 500

        theta_real = np.array([4.0, -5.3, 2.2, 11.8, 6.4, 0.8])
        X = np.random.rand(500, 6)
        y = X @ theta_real

        def grad(theta, aux_data): return lf.squared_error_cost_gradient(aux_data[0], aux_data[1], theta)

        start = np.zeros(6)

        theta = None
        for theta in opt.gradient_descend(grad_fn=grad, data_stream=opt.mini_batch_stream(X, y, 32, iters), alpha=.01,
                                          init_theta=start):
            pass

        assert(fuzzy_equals(theta, theta_real, .1))


if __name__ == "__main__":
    unittest.main()
