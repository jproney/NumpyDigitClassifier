import neural_net as nn
import loss_functions as lf
import optimizers as opt
import numpy as np
import utils

net = nn.NeuralNet()
net.add_layer(nn.WeightLayer(weights=np.random.rand(3, 5), biases=np.random.rand(5)))
net.add_layer(nn.WeightLayer(weights=np.random.rand(5, 3), biases=np.random.rand(3)))

X = np.array([[1,2,3],[3,1,2],[3,2,1]])
Y = np.array([[1,0,0],[0,1,0],[0,0,1]])


def grad(network, data): return network.backpropagate(data[0], data[1], lf.cross_entropy_cost_gradient)


iters = 10000
for Theta in opt.gradient_descend(grad_fn=grad, data_stream=opt.full_batch_stream(X, Y, iters),
                                  alpha=.01, init_theta=net, update_fn=utils.update_neural_net):
    pass

print(lf.softmax(net.compute(X)[-1]))
