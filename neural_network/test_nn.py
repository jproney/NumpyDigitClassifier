import neural_net as nn
import loss_functions as lf
import numpy as np

net = nn.NeuralNet()
net.add_layer(nn.WeightLayer(weights=np.random.rand(3, 5), biases=np.random.rand(5)))
net.add_layer(nn.WeightLayer(weights=np.random.rand(5, 3), biases=np.random.rand(3)))
for i in range(10000):
    net.backpropogate_and_update(np.array([[1,2,3],[3,1,2],[3,2,1]]), np.array([[1,0,0],[0,1,0],[0,0,1]]), lf.cross_entropy_cost_gradient, alpha=.01)
print(net.compute(np.array([[1,2,3],[3,1,2],[3,2,1]])))