#Nonlinear activation functions for neural net


def relu(x): return (x > 0)*x


def relu_prime(x): return x > 0
