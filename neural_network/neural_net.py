import numpy as np


def relu(x): 0 if x < 0 else x


def relu_prime(x): 0 if x < 0 else 1


class WeightLayer:

    def __init__(self, num_inputs, num_outputs, activ=relu, active_prime=relu_prime):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activ = activ
        self.activ_prime = activ_prime
        self.weight_matrix = np.random.rand((num_outputs, num_inputs + 1))

    def compute(self, X):
        with_bias = np.vstack((input_vec, np.ones((X.shape[1], 1)))))
        return self.activ(self.weight_matrix @ X)

    def set_weights(self, new):
        self.weights = new

    def compute_activations_gradient(self, X)  # gradient of outputs wrt each input
        with_bias = np.vstack((input_vec, np.ones((X.shape[1], 1)))))
        h = self.weights @ with_bias
        derivs = self.active_prime(h)
        return weights.T[:-1, :] * derivs  # matrix where [x,y] = d(ouput y)/d(input x) (excludes bias input)

    def compute_weights_gradient(self, X):  # gradient of outputs wrt each weight
        with_bias = np.vstack((input_vec, np.ones((X.shape[1], 1)))))
        h = self.weights @ with_bias
        derivs = self.active_prime(h)
        return (with_bias @ derivs.T) / X.shape[1]  # matrix where [x,y] = d(output y)/d(weight x)


class NeuralNet():

    def __init__(self):
        self.weight_layers = []
        self.num_layers = 0

    def add_layer(layer):
        self.weight_layers.append(layer)
        self.num_layers += 1

    def compute(self, X):
        layer_outputs = [X]
        for wl in self.weight_layers:
            layer_outputs.append(wl.compute(layer_outputs[-1]))
        return layer_outputs

    def backpropogate(self, X, cost_gradient)
        activation_grads = []
        weight_grads = []
        outputs = self.compute(X)
        activation_grads.append(cost_gradient(X))  # cost wrt activations of output layer
        for i, wl in reversed(self.weight_layers):
            weight_grads.append(wl.compute_weights_gradient(outputs[i - 1]) @ activation_grads[-1])
            activation_grads.append(compute_activations_gradient(X) @ activation_grads[-1])
