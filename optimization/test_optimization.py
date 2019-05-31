import time
import unittest

from optimization import optimizers as opt
from optimization import loss_functions as lf
from optimization import utils
import matplotlib.pyplot as plt
import mnist
import numpy as np


class LinearTestCase(unittest.TestCase):

    def test_basic_GD(self):
        print("testing gradient descent...")
        b = np.random.rand(3, 1) * 20
        noise = np.random.rand(10000, 1) * 20000
        input_vec = np.random.rand(10000, 1) * 100
        train_ex = np.hstack((input_vec ** 2, input_vec, np.ones((10000, 1))))
        sol = train_ex @ b + noise
        start = time.time()
        (train_ex, (in_add, in_scale)) = utils.column_normalize(train_ex)
        (sol, (out_add, out_scale)) = utils.column_normalize(sol)
        (theta, err) = opt.gradient_descent(train_ex, sol, lf.squared_error_cost_gradient, epochs=200, track_err=True,
                                            error_fn=lf.squared_error_cost)
        print("Converged in {} Seconds".format(time.time() - start))
        train_ex.sort(0)
        h = train_ex @ theta
        plt.subplot(2, 1, 1)
        plt.plot(input_vec, sol, 'bo', linestyle='none')
        input_vec.sort(0)
        plt.plot(input_vec, h, color='r', linewidth=4)
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(err.shape[0]), err)
        plt.show()

    def test_SGD(self):
        print("testing SGD...")
        b = np.random.rand(3, 1) * 20
        noise = np.random.rand(10000, 1) * 20000
        input_vec = np.random.rand(10000, 1) * 100
        train_ex = np.hstack((input_vec ** 2, input_vec, np.ones((10000, 1))))
        sol = train_ex @ b + noise
        start = time.time()
        (train_ex, (in_add, in_scale)) = utils.column_normalize(train_ex)
        (sol, (out_add, out_scale)) = utils.column_normalize(sol)
        (theta, err) = opt.sgd_optimize(train_ex, sol, lf.squared_error_cost_gradient, 32, epochs=100, track_err=True,
                                        error_fn=lf.squared_error_cost)
        print("Converged in {} Seconds".format(time.time() - start))
        train_ex.sort(0)
        h = train_ex @ theta
        plt.subplot(2, 1, 1)
        plt.plot(input_vec, sol, 'bo', linestyle='none')
        input_vec.sort(0)
        plt.plot(input_vec, h, color='r', linewidth=4)
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(err.shape[0]), err)
        plt.show()

    def test_adam(self):
        print("testing ADAM...")
        b = np.random.rand(3, 1) * 20
        noise = np.random.rand(10000, 1) * 20000
        input_vec = np.random.rand(10000, 1) * 100
        train_ex = np.hstack((input_vec ** 2, input_vec, np.ones((10000, 1))))
        sol = train_ex @ b + noise
        start = time.time()
        (train_ex, (in_add, in_scale)) = utils.column_normalize(train_ex)
        (sol, (out_add, out_scale)) = utils.column_normalize(sol)
        (theta, err) = opt.adam_optimize(train_ex, sol, lf.squared_error_cost_gradient, 32, epochs=100, track_err=True,
                                         error_fn=lf.squared_error_cost)
        print("Converged in {} Seconds".format(time.time() - start))
        train_ex.sort(0)
        h = train_ex @ theta
        plt.subplot(2, 1, 1)
        plt.plot(input_vec, sol, 'bo', linestyle='none')
        input_vec.sort(0)
        plt.plot(input_vec, h, color='r', linewidth=4)
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(err.shape[0]), err)
        plt.show()

    def test_logistic(self):
        print("testing logistic regression on MNIST...")
        m = mnist.MNIST('/home/petey/Documents/PythonProjects/MachineLearning/python-mnist/data')
        images, labels = m.load_training()
        train_ex = np.asarray(images)
        (train_ex, (in_add, in_scale)) = utils.column_normalize(train_ex)
        train_ex = np.hstack((np.ones((train_ex.shape[0], 1)), train_ex))
        labels = np.tile(np.asarray(labels), (10, 1)).T
        cats = np.tile(np.arange(0, 10), (train_ex.shape[0], 1))
        sol = np.equal(labels, cats).astype('float64')  # create on-hot vector
        theta0 = np.zeros((train_ex.shape[1], sol.shape[1]))
        (theta, err) = opt.sgd_optimize(train_ex, sol, lf.cross_entropy_cost_gradient, 1, epochs=10, alpha=.001,
                                        init_theta=theta0, track_err=True, error_fn=utils.softmax_classification_accuracy,
                                        track_progress=True)
        plt.plot(np.arange(err.shape[0]), err)
        print(np.min(err))
        plt.show()


if __name__ == "__main__":
    unittest.main()
