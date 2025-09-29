import math
import numpy as np

def train_test_split(X_all, T_all, train_size = 0.8, shuffle = True):
    split_N = math.trunc(len(X_all) * train_size)
    
    X, T = X_all, T_all
    if shuffle:
        p = np.random.permutation(len(X_all))
        X, T = X_all[p], T_all[p]
    
    X_train = X[: split_N]
    T_train = T[: split_N]
    X_eval = X[split_N: ]
    T_eval = T[split_N: ]
    
    return X_train, X_eval, T_train, T_eval

# Weight and bias init functions
def zeros_init(n_in, n_out):
    return np.zeros((n_in, n_out), dtype=float)

def ones_init(n_in, n_out):
    return np.ones((n_in, n_out), dtype=float)

def arrange_init(n_in, n_out):
    return np.arange(1, n_in * n_out+1, dtype=float).reshape(n_in, n_out)

def rand_uniform_init(n_in, n_out):
    return np.random.uniform(-0.1, 0.1, (n_in, n_out))

def rand_norm_init(n_in, n_out):
    return np.random.normal(0, 0.1, (n_in, n_out))

# Better for ReLU
def He_init(n_in, n_out):
    return np.random.normal(0, math.sqrt(2 / n_in), (n_in, n_out))

# Better for tanh
def Xavier(n_in, n_out):
    return math.sqrt(6/(n_in+n_out))

def Xavier_init(n_in, n_out):
    return np.random.uniform(-Xavier(n_in, n_out), Xavier(n_in, n_out), (n_in, n_out))

def softplus(X):
    # stable: log(1 + exp(x))
    return np.log1p(np.exp(-np.abs(X))) + np.maximum(X, 0)
