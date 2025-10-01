import random
import numpy as np
from collections import namedtuple
import copy
from utils import *
from activation_functions import *

class mse:
    def __init__(self, layers):
        self.layers = layers

    def loss(self, T, Y, Z):
        # common in ML: mean over batch, sum over outputs
        # should be consistent with dL_dA = (A[-1] - T) / len(X)
        return np.mean(1/2 * np.sum((T - Y)**2, axis=1))

    def dL_dZ(self, T, A, Z):
        B = len(T)
        dL_dA = (A[-1] - T) / B  # scale 1/B (mean over batch), (B, n_out)
        return dL_dA * self.layers[-1].a_fn_prime(A[-1])

# Binary cross-entropy uses logits only
class bce:
    def loss(self, T, Y, Z):
        return np.mean(softplus(Z) - T * Z)

    def dL_dZ(self, T, A, Z):
        B = len(T)
        return (sigmoid(Z[-1]) - T) / B

class bce_weighted:
    def __init__(self, pos_weight=1.0, neg_weight=1.0):
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def loss(self, T, Y, Z):
        return np.mean(
            self.neg_weight * (1.0 - T) * softplus(Z) +
            self.pos_weight * T * (softplus(Z) - Z))

    def dL_dZ(self, T, A, Z):
        B = len(T)
        s = sigmoid(Z[-1])
        return (self.neg_weight * (1.0 - T) * s + self.pos_weight * T * (s - 1.0)) / B

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
    def __init__(self, w_init, b_init, layers, alg):
        assert len(layers) >= 2, "Min amount of layers: input, output"
        for layer in layers[1:]:
            assert layer.a_fn != None and layer.a_fn_prime != None, "Every layer exept input should have defined activation function"

        self.alg = alg
        self.layers = layers
        self.W = [w_init(layerN_minus1.neurons, layerN.neurons) for layerN_minus1, layerN in zip(layers[:-1], layers[1:])]
        self.b = [b_init(1, layer.neurons).flatten() for layer in layers[1:]]
    
        self.best_W = []
        self.best_b = []

    def forward(self, X):
        assert X.shape[1] == self.layers[0].neurons, "X must have shape (batch_size, n_features) with n_features == input layer size"
        Z = []
        A = []
        A.append(X)

        for b, W, layer in zip(self.b, self.W, self.layers[1:]):
            Z.append( A[-1] @ W + b )
            A.append( layer.a_fn(Z[-1]) )

        A.pop(0)

        return namedtuple("Y_Z_A", "Y Z A")(A[-1], Z, A)

    def gradients(self, X, Z, A, T):
        B = len(X)

        dW = []
        db = []
        dZ = []

        A.insert(0, X)
        
        dL_dZ = self.alg.dL_dZ(T, A, Z)

        for i in range(len(self.layers[1:]) - 1, -1, -1):
            dZ_dW = A[i]
            
            dL_dW = dZ_dW.T @ dL_dZ # (n_in, n_out)
            dL_db = dL_dZ.sum(axis=0)   # (n_out,)
            
            dW.insert(0, dL_dW)
            db.insert(0, dL_db)
            dZ.insert(0, dL_dZ)
            
            if i > 0:
                # prepare for next interation
                dL_dA = dL_dZ @ self.W[i].T  # (B, n_out) @ (n_out, n_in) -> (B, n_in)
                dA_dZ = self.layers[i].a_fn_prime(A[i])   # (B, n_out)
                dL_dZ = dL_dA * dA_dZ   # multiply elementwise, (B, n_out)

        A.pop(0)

        return namedtuple("dW_db_dZ", "dW db dZ")(dW, db, dZ)
    
    def update_W_b(self, dW, db, eta):
        for i in range(len(self.W)):
            self.W[i] -= eta * dW[i]
            self.b[i] -= eta * db[i]

    def remember_best_state(self):
        self.best_W = copy.deepcopy(self.W)
        self.best_b = copy.deepcopy(self.b)

    def revert_to_best_state(self):
        if self.best_W and self.best_b:
            self.W = copy.deepcopy(self.best_W)
            self.b = copy.deepcopy(self.best_b)
                
    def train(self, X_train, X_eval, T_train, T_eval, max_epochs, batch_size, eta, eta_decay_rate = 0.98, patience=50, rel_tol=3e-3):
        assert len(X_train) == len(T_train) and len(X_eval) == len(T_eval), "Size of X and T should be the same"
        assert batch_size <= len(X_train), "batch_size should be smaller or same size as input X"
        if isinstance(self.alg, (bce, bce_weighted)):
            assert T_train.shape[1] == 1 and set(np.unique(T_train)).issubset({0,1}), "Binary targets must be shape (B,1) with values 0/1"

        train_log = []
        
        # Patience params
        min_epochs = 200        
        eps = 1e-12 # for numerical stability in relative test
        L_eval_best = 1e+9 # set too far at the beggining
        stale = 0

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
                L_train += self.alg.loss(T_batch, Y_batch, Z_batch[-1]) * len(X_batch)

            # average over all mini-batches
            L_train /= len(X_train)
            
            # Compute eval loss
            Y_eval, Z_eval, _ = self.forward(X_eval)
            L_eval =  self.alg.loss(T_eval, Y_eval, Z_eval[-1])
            
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