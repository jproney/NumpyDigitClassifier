import neural_net as nn
import loss_functions as lf
import numpy as np
import optimizers as opt
import utils
import matplotlib.pyplot as plt
import mnist


# load the data
m = mnist.MNIST('/home/petey/Documents/PythonProjects/MachineLearning/python-mnist/data')
images, labels = m.load_training()
X = np.array(images)
X_norm, mean, std, keep = utils.column_normalize(X)
y = np.array(labels)
K = 10
Y = utils.to_one_hot(y, K)

# make the network
hidden_size = 1000
net = nn.NeuralNet()
net.add_layer(nn.WeightLayer(weights=np.random.rand(X_norm.shape[1], hidden_size)*.1 - .05, biases=np.random.rand(hidden_size)*.1 - .05))
net.add_layer(nn.WeightLayer(weights=np.random.rand(hidden_size, K)*.1 - .05, biases=np.random.rand(K)*.1-.05))


# helper functions

def grad(network, data): return network.backpropagate(data[0], data[1], lf.cross_entropy_cost_gradient)


# train!
epochs = 10
counter = 0
losslog = []

for Theta in opt.gradient_descend(grad_fn=grad, data_stream=opt.mini_batch_stream(X_norm, Y, 4, epochs),
                                  alpha=.001, init_theta=net, update_fn=utils.update_neural_net):
    if counter % 1000 == 0:
        losslog.append(lf.cross_entropy_cost(net.compute(X_norm)[-1], Y))
        print(counter)
    counter += 1

plt.plot(losslog)
plt.show()
print("Training set error rate:")
print(utils.softmax_classification_accuracy(net.compute(X_norm)[-1], Y))

im_test, lab_test = m.load_testing()
X_test = utils.column_normalize_with_params(np.array(im_test), keep, mean, std)
Y_test = utils.to_one_hot(np.array(lab_test), K)
print("Test set error rate:")
print(utils.softmax_classification_accuracy(net.compute(X_test)[-1], Y_test))