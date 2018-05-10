import unittest
import gradient_descent as gd
import numpy as np
import random
import matplotlib.pyplot as plt

def fuzzy_equals(a,b):
    return np.all(np.less(a-b,np.ones(a.shape)*.001))

class LinearTestCase(unittest.TestCase):
    
    def testA(self):
        b = np.array([random.randrange(0,20) for i in range(3)])
        noise = np.random.rand(100,1)*50
        train_ex = np.array([[x**2 ,x ,1] for x in range(0,100)])
        sol = np.ndarray.flatten(train_ex @ b) #+ np.ndarray.flatten(noise)
        theta = gd.gradient_descent(train_ex,sol, gd.linear_squared_error_gradient)
        print(theta)
        assert fuzzy_equals(theta, b), "error outside tolerance"
        
if __name__ == "__main__":
    unittest.main()

