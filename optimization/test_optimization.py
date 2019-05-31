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

if __name__ == "__main__":
    unittest.main()
