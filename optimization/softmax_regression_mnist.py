import numpy as np
from optimization import loss_functions as lf
from optimization import optimizers as opt
from optimization import utils
import mnist

m = mnist.MNIST('/home/petey/Documents/PythonProjects/MachineLearning/python-mnist/data')
images, labels = m.load_training()
X = np.array(images)
X_norm, mean, std, keep = utils.column_normalize(X)
y = np.array(labels)
K = 10
Y = utils.to_one_hot(y, K)
Theta_init = np.zeros((X_norm.shape[1], K))


def grad(Theta, aux_data): return lf.cross_entropy_cost_gradient(aux_data[0], aux_data[1], Theta)


epochs = 150000
Theta = None
i = 0
for (i, Theta) in zip(range(epochs), opt.gradient_descend(grad_fn=grad, data_stream=opt.mini_batch_stream(X_norm, Y, 1, epochs), alpha=.001,
                                                          init_theta=Theta_init)):
    print(i)
    pass


print(utils.softmax_classification_accuracy(X_norm, Y, Theta))