"""
All computations are vectorized to handle multiple examples at once. Every data matrix has N rows,
where N is the number of examples. Weight matrices are dimensioned accordingly, resulting
in matrix multiplications which are transposed from certain other implementations.
"""
import numpy as np
import activations as act


class WeightLayer:

    def __init__(self, weights, activ=act.relu, activ_prime=act.relu_prime):
        # weights[i,j] is a weight mapping from neuron i in layer l to j in layer l+1
        self.num_inputs = weights.shape[0]
        self.num_outputs = weights.shape[1]
        self.activ = activ
        self.activ_prime = activ_prime
        self.weight_matrix = weights

    def compute(self, A):  # matrix with inputs from different examples stacked vertically
        with_bias = np.hstack((A, np.ones(A.shape[1])))
        return self.activ(with_bias @ self.weight_matrix)

    def set_weights(self, new):
        self.weight_matrix = new

    def get_weights(self):
        return self.weight_matrix


class NeuralNet:

    def __init__(self):
        self.weight_layers = []
        self.num_layers = 0

    def add_layer(self, layer):
        self.weight_layers.append(layer)
        self.num_layers += 1

    def compute(self, X):
        activations = [X]
        curr_input = X
        for wl in self.weight_layers:
            curr_input = wl.compute(curr_input)
            activations.append(curr_input)
        return activations

    def backpropogate(self, X, Y, cost_gradient):
        X_out = self.compute(X)  # activations of all layers
        Delta = [cost_gradient(X_out[-1], Y)]  # N x K matrix
        Grad = [X[-2].T @ Delta[0]]
        for i, wl in zip(reversed(range(self.num_layers, 1)), reversed(self.weight_layers)):  # iterate backwards to second layer
            Z = X[i-1] @ wl.weights
            delta_prev = (Delta[0] @ wl.weights.T) * Z
            Delta.insert(0, delta_prev)
            grad = X[i-1].T @ Delta[0]
            Grad.insert(0, grad)
        return Grad



