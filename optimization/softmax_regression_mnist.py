import numpy as np
import loss_functions as lf
import optimizers as opt
import utils
import matplotlib.pyplot as plt
import mnist

m = mnist.MNIST('/home/petey/Documents/PythonProjects/MachineLearning/python-mnist/data')
images, labels = m.load_training()
X = np.array(images)
X_norm, mean, std, keep = utils.column_normalize(X)
y = np.array(labels)
K = 10
Y = utils.to_one_hot(y, K)
Theta_init = np.zeros((X_norm.shape[1], K))


def grad(Theta, aux_data): return lf.cross_entropy_cost_gradient_theta(aux_data[0], aux_data[1], Theta)


epochs = 20
counter = 0
Theta = None
losslog = []

for Theta in opt.gradient_descend(grad_fn=grad, data_stream=opt.mini_batch_stream(X_norm, Y, 4, epochs),
                                  alpha=.001, init_theta=Theta_init):
    if counter % 1000 == 0:
        losslog.append(lf.cross_entropy_cost_theta(X_norm, Y, Theta))
        print(counter)
    counter += 1

plt.plot(losslog)
plt.show()
print(utils.softmax_classification_accuracy_theta(X_norm, Y, Theta))

