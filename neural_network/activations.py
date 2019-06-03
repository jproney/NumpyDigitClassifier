#Nonlinear activation functions for neural net


def relu(x): return 0 if x < 0 else x


def relu_prime(x): return 0 if x < 0 else 1
