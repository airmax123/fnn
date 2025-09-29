import numpy as np

# Activation functions
# For prime - use post-activation A, to avoid double calculations
def identity(X):
    return X

def identity_prime(A):
    return np.ones_like(A)

def tanh(X):
    return np.tanh(X)

def tanh_prime(A):
    return 1 - A**2

def LeakyReLU(X, alpha=0.01):
    return np.where(X > 0, X, alpha * X)

def LeakyReLU_prime(A, alpha=0.01):
    return np.where(A > 0, 1.0, alpha)

def sigmoid(X):
    out = np.empty_like(X)
    pos = X >= 0
    out[pos]  = 1.0 / (1.0 + np.exp(-X[pos]))
    expx = np.exp(X[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out

def sigmoid_prime(A):
    return A * (1.0 - A)
