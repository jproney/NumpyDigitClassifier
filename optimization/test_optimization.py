import unittest
import gradient_descent as gd
import numpy as np

def fuzzy_equals(a,b):
    return np.all(np.less(a-b,np.ones(a.shape)*.01))

class LinearTestCase(unittest.TestCase):
    
    def testA(self):
        train_ex = np.array([[1,1,1],[1,2,4],[1,3,9]])
        sol = np.array([[3],[8],[15]])
        theta = gd.gradient_descent(train_ex,sol)
        print(theta)
        assert fuzzy_equals(theta, np.array([[0],[2],[1]])), "error outside tolerance"
        
if __name__ == "__main__":
    unittest.main()

