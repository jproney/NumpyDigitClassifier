import unittest
import gradient_descent as gd
import numpy as np
import random
import matplotlib.pyplot as plt
import time

def fuzzy_equals(a,b):
    return np.all(np.less(a-b,np.ones(a.shape)*.001))

class LinearTestCase(unittest.TestCase):
    
    def testA(self):
        b = np.random.rand(3,1)*20
        noise = np.random.rand(500,1)*20000
        input_vec = np.random.rand(500,1)*100
        train_ex = np.hstack((input_vec**2,input_vec,np.ones((500,1))))
        sol = np.ndarray.flatten(train_ex @ b) + np.ndarray.flatten(noise)
        start = time.time()
        theta = gd.gradient_descent(train_ex,sol, gd.linear_squared_error_gradient)
        print("Converged in {} Seconds".format(time.time()-start))
        train_ex.sort(0)
        (te_norm,_) = gd.column_normalize(train_ex)
        h = te_norm @ theta
        plt.plot(input_vec,sol,'bo',linestyle='none')
        input_vec.sort(0)
        plt.plot(input_vec,h,color='r')
        plt.show()
        
if __name__ == "__main__":
    unittest.main()

