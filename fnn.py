import random
import math
import numpy as np
from collections import namedtuple
import copy

def as_column_vector(V):
    return np.asarray(V)[:, np.newaxis]

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

# Loss
def loss_mean(T, Y):
    L_per_sample = 1/2 * (T - Y)**2
    return np.mean(L_per_sample)

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

# Feeedforward neural network
class Layer:
    def __init__(self, neurons, a_fn, a_fn_prime):
        self.neurons = neurons
        self.a_fn = a_fn
        self.a_fn_prime = a_fn_prime

# Batch-first network layout (PyTorch, TensorFlow, scikit-learn, NumPy) 
# X: (batch_size, n_features_in)
# Y: (batch_size, n_outputs)
# W: (n_in, n_out)
# b: (n_out,) <- 1-D vector. In such way NumPy broadcasts (n_out,) across rows
# In such case math is: Z = A_prev @ W + b (in opposite to math formula Z = W @ A_prev + b)
class Fnn:
    def __init__(self, w_init, b_init, layers):
        assert len(layers) >= 2, "Min amount of layers: input, output"
        for layer in layers[1:]:
            assert layer.a_fn != None and layer.a_fn_prime != None, "Every layer exept input should have defined activation function"

        self.layers = layers
        self.W = [w_init(layerN_minus1.neurons, layerN.neurons) for layerN_minus1, layerN in zip(layers[:-1], layers[1:])]
        self.b = [b_init(1, layer.neurons).flatten() for layer in layers[1:]]
    
        self.best_W = []
        self.best_b = []

    def forward(self, X):
        assert X.shape[1] == self.layers[0].neurons, "X should be column vector"
        Z = []
        A = []
        A.append(X)

        for b, W, layer in zip(self.b, self.W, self.layers[1:]):
            Z.append( A[-1] @ W + b )
            A.append( layer.a_fn(Z[-1]) )

        A.pop(0)

        return namedtuple("Y_Z_A", "Y Z A")(A[-1], Z, A)

    def gradients(self, X, Z, A, T):
        dW = []
        db = []
        dZ = []

        A.insert(0, X)
        dL_dA = (A[-1] - T) / len(X)  # scale 1/B (mean over batch), (B, n_out)
        
        for i in range(len(self.layers[1:]) - 1, -1, -1):
            a_prime = self.layers[i + 1].a_fn_prime

            dA_dZ = a_prime(A[i + 1])   # (B, n_out) 
            dL_dZ = dL_dA * dA_dZ   # multiply elementwise, (B, n_out)
    
            dZ_dW = A[i]
            #dZ_db = np.array([1])
            
            dL_dW = dZ_dW.T @ dL_dZ # (n_in, n_out)
            dL_db = dL_dZ.sum(axis=0)   # (n_out,)
            
            dL_dA = dL_dZ @ self.W[i].T  # (B, n_out) @ (n_out, n_in) -> (B, n_in)

            dW.insert(0, dL_dW)
            db.insert(0, dL_db)
            dZ.insert(0, dL_dZ)

        A.pop(0)

        return namedtuple("dW_db_dZ", "dW db dZ")(dW, db, dZ)
    
    def update_W_b(self, dW, db, eta):
        for i in range(len(self.W) - 1):
            self.W[i] -= eta * dW[i]
            self.b[i] -= eta * db[i]

    def remember_best_state(self):
        self.best_W = copy.deepcopy(self.W)
        self.best_b = copy.deepcopy(self.b)

    def revert_to_best_state(self):
        if self.best_W and self.best_b:
            self.W = copy.deepcopy(self.best_W)
            self.b = copy.deepcopy(self.best_b)
                
    def train(self, X_all, T_all, max_epochs, batch_size, eta, eta_decay_rate = 0.98):
        assert len(X_all) == len(T_all), "Size of X and T should be the same"
        assert batch_size <= len(X_all), "batch_size should be smaller or same size as input X"
        
        train_log = []
        
        # Patience params
        min_epochs = 200        
        rel_tol = 1e-3 # relative improvement threshold
        patience = 10 # epochs without improvement before stop
        eps = 1e-12 # for numerical stability in relative test
        L_eval_best = 1e+9 # set too far at the beggining
        stale = 0

        # Shuffle data
        p = np.random.permutation(len(X_all))
        X_shuffled, T_shuffled = X_all[p], T_all[p]
        
        # Split data 80/20
        split_N = math.trunc(len(X_all) * 0.8)
        X_train = X_shuffled[: split_N]
        T_train = T_shuffled[: split_N]
        X_eval = X_shuffled[split_N: ]
        T_eval = T_shuffled[split_N: ]

        if batch_size == len(X_all) or batch_size >= split_N:
            batch_size = split_N

        for epoch in range(max_epochs):
            p = np.random.permutation(len(X_train))
            X, T = X_train[p], T_train[p]

            X_batched = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
            T_batched = [T[i:i + batch_size] for i in range(0, len(T), batch_size)]
            
            L_train = 0        
            for X_batch, T_batch in zip(X_batched, T_batched):
                Y_batch, Z_batch, A_batch = self.forward(X_batch)
                dW, db, _ = self.gradients(X_batch, Z_batch, A_batch, T_batch)
                self.update_W_b(dW, db, eta)
                L_train += loss_mean(T_batch, Y_batch) * len(X_batch)
            
            # average over all mini-batches
            L_train /= len(X_train)
            
            # Compute eval loss
            L_eval = loss_mean(T_eval, self.forward(X_eval).Y)
            
            # Relative improvement test
            impr = (L_eval_best - L_eval) / (abs(L_eval_best) + eps)
            if impr > rel_tol:            
                self.remember_best_state()
                L_eval_best = L_eval
                stale = 0
            else:
                stale += 1
            
            train_log += [[L_train, L_eval, stale]]
            
            if epoch >= min_epochs and stale >= patience:
                # we'll restore best state on exit from the loop
                break

            eta = eta * eta_decay_rate
        
        self.revert_to_best_state()
        
        return np.array(train_log)