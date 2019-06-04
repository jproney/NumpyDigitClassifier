"""
All computations are vectorized to handle multiple examples at once. Every data matrix has N rows,
where N is the number of examples. Weight matrices are dimensioned accordingly, resulting
in matrix multiplications which are transposed from certain other implementations.
"""
import numpy as np
import activations as act


class WeightLayer:

    def __init__(self, weights, biases, activ=act.relu, activ_prime=act.relu_prime):
        # weights[i,j] is a weight mapping from neuron i in layer l to j in layer l+1

        self.num_inputs = weights.shape[0]
        self.num_outputs = weights.shape[1]
        self.activ = activ
        self.activ_prime = activ_prime
        self.weights = weights
        self.biases = biases

    def compute(self, A):  # matrix with inputs from different examples stacked vertically
        return self.activ(A @ self.weights + self.biases)

    def set_weights(self, new):
        self.weights = new

    def get_weights(self):
        return self.weights


class NeuralNet:

    def __init__(self):
        self.layers = []
        self.num_layers = 0  # input layer by default

    def add_layer(self, layer):
        self.layers.append(layer)
        self.num_layers += 1

    def compute(self, X):
        activations = np.empty(self.num_layers + 1, dtype=object)
        activations[0] = X
        curr_input = X
        for l in range(self.num_layers):
            curr_input = self.layers[l].compute(curr_input)
            activations[l+1] = curr_input
        return activations

    def backpropagate(self, X, Y, cost_gradient):
        A = self.compute(X)  # activations of all layers
        Delta = np.empty(self.num_layers, dtype=object)
        Grad = np.empty(self.num_layers, dtype=object)

        Delta[self.num_layers-1] = cost_gradient(A[-1], Y)  # N x K matrix Delta_L
        Grad[self.num_layers-1] = A[-2].T @ Delta[-1]

        for l in reversed(range(1, self.num_layers)):  # iterate backwards to second layer
            Z_l = A[l-1] @ self.layers[l-1].weights + self.layers[l-1].biases
            Delta[l-1] = (Delta[l] @ self.layers[l].weights.T) * self.layers[l].activ_prime(Z_l)
            Grad[l-1] = A[l-1].T @ Delta[l-1]  # delta indices are down-shifted by 1

        Grad_bias = np.array([np.sum(delt, axis=0) for delt in Delta])
        return np.concatenate((Grad, Grad_bias))

    def backpropagate_and_update(self, X, Y, cost_gradient, alpha):
        Grad = self.backpropagate(X, Y, cost_gradient)
        for l in range(self.num_layers):
            self.layers[l].weights -= alpha*Grad[l]
            self.layers[l].biases -= alpha*Grad[self.num_layers + l]
