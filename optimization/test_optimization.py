import unittest
import gradient_descent as gd
import numpy as np
import random
import matplotlib.pyplot as plt
import time

def fuzzy_equals(a,b):
    return np.all(np.less(a-b,np.ones(a.shape)*.001))

class LinearTestCase(unittest.TestCase):
    
    def testBasicGD(self):
        print("testing gradient descent...")
        b = np.random.rand(3,1)*20
        noise = np.random.rand(10000,1)*20000
        input_vec = np.random.rand(10000,1)*100
        train_ex = np.hstack((input_vec**2,input_vec,np.ones((10000,1))))
        sol = np.ndarray.flatten(train_ex @ b) + np.ndarray.flatten(noise)
        start = time.time()
        (theta,(in_add,in_scale),(out_add,out_scale),err) = gd.gradient_descent(train_ex,sol, gd.linear_squared_error_gradient,iterations = 200,track_err = True)
        print("Converged in {} Seconds".format(time.time()-start))
        train_ex.sort(0)
        te_norm = np.subtract(train_ex, in_add)/in_scale
        h = np.add(np.multiply(te_norm @ theta,out_scale),out_add)
        plt.subplot(2,1,1)
        plt.plot(input_vec,sol,'bo',linestyle='none')
        input_vec.sort(0)
        plt.plot(input_vec,h,color='r',linewidth=4)
        plt.subplot(2,1,2)
        plt.plot(np.arange(err.shape[0]),err)
        plt.show()


    def testSGD(self):
        print("testing SGD...")
        b = np.random.rand(3,1)*20
        noise = np.random.rand(10000,1)*20000
        input_vec = np.random.rand(10000,1)*100
        train_ex = np.hstack((input_vec**2,input_vec,np.ones((10000,1))))
        sol = np.ndarray.flatten(train_ex @ b) + np.ndarray.flatten(noise)
        start = time.time()
        (theta,(in_add,in_scale),(out_add,out_scale),err) = gd.sgd_optimize(train_ex,sol, gd.linear_squared_error_gradient, 1000, track_err = True)
        print("Converged in {} Seconds".format(time.time()-start))
        train_ex.sort(0)
        te_norm = np.subtract(train_ex, in_add)/in_scale
        h = np.add(np.multiply(te_norm @ theta,out_scale),out_add) 
        plt.subplot(2,1,1)
        plt.plot(input_vec,sol,'bo',linestyle='none')
        input_vec.sort(0)
        plt.plot(input_vec,h,color='r',linewidth=4)
        plt.subplot(2,1,2)
        plt.plot(np.arange(err.shape[0]),err)
        plt.show()


    def testAdam(self):
        print("testing ADAM...")
        b = np.random.rand(3,1)*20
        noise = np.random.rand(10000,1)*20000
        input_vec = np.random.rand(10000,1)*100
        train_ex = np.hstack((input_vec**2,input_vec,np.ones((10000,1))))
        sol = np.ndarray.flatten(train_ex @ b) + np.ndarray.flatten(noise)
        start = time.time()
        (theta,(in_add,in_scale),(out_add,out_scale),err) = gd.adam_optimize(train_ex,sol, gd.linear_squared_error_gradient, 1000, track_err = True)
        print("Converged in {} Seconds".format(time.time()-start))
        train_ex.sort(0)
        te_norm = np.subtract(train_ex, in_add)/in_scale
        h = np.add(np.multiply(te_norm @ theta,out_scale),out_add) 
        plt.subplot(2,1,1)
        plt.plot(input_vec,sol,'bo',linestyle='none')
        input_vec.sort(0)
        plt.plot(input_vec,h,color='r',linewidth=4)
        plt.subplot(2,1,2)
        plt.plot(np.arange(err.shape[0]),err)
        plt.show()
        
if __name__ == "__main__":
    unittest.main()

