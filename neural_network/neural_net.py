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
        activations = [X]
        curr_input = X
        for wl in self.layers:
            curr_input = wl.compute(curr_input)
            activations.append(curr_input)
        return activations

    def backpropogate(self, X, Y, cost_gradient):
        A = self.compute(X)  # activations of all layers
        Delta = [cost_gradient(A[-1], Y)]  # N x K matrix Delta_L
        Grad = [A[-2].T @ Delta[0]]

        for l in reversed(range(1, self.num_layers)):  # iterate backwards to second layer
            Z_l = A[l-1] @ self.layers[l-1].weights + self.layers[l-1].biases
            delta_prev = (Delta[0] @ self.layers[l].weights.T) * self.layers[l].activ_prime(Z_l)
            Delta.insert(0, delta_prev)
            grad = A[l-1].T @ Delta[0]
            Grad.insert(0, grad)

        Grad_bias = [np.sum(delt, axis=0) for delt in Delta]
        return Grad, Grad_bias

    def backpropogate_and_update(self, X, Y, cost_gradient, alpha):
        Grad, Grad_bias = self.backpropogate(X,Y,cost_gradient)
        for l in range(self.num_layers):
            self.layers[l].weights -= alpha*Grad[l]
            self.layers[l].biases -= alpha*Grad_bias[l]
