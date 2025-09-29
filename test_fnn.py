import random
import math
import numpy as np
import unittest
from collections import namedtuple
import copy
from fnn import *

def mse_loss_mean(T, Y):
    return mse(None).loss(T, Y, None)

def bce_loss_mean(T, Y):
    return bce().loss(T, Y, Y)
    

# UnitTests
class TestFnn(unittest.TestCase):
    def test_fnn_mse_loss_mean(self):
        Y = as_column_vector([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        T = as_column_vector([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        
        L = mse_loss_mean(T, Y)

        self.assertEqual(L, 0.0)

    def test_fnn_W_b_shape(self):
        features_X = 2
        neurons_L1 = 3
        neurons_L2 = 4
        features_Y = 1
        fnn_layers = [Layer(features_X, None, None),
                      Layer(neurons_L1, identity, identity_prime), 
                      Layer(neurons_L2, identity, identity_prime),
                      Layer(features_Y, identity, identity_prime)]

        fnn = Fnn(w_init = zeros_init, b_init = ones_init, layers = fnn_layers, alg = mse(fnn_layers))

        self.assertEqual(fnn.W[0].shape, (features_X, neurons_L1))
        self.assertEqual(fnn.W[1].shape, (neurons_L1, neurons_L2))
        self.assertEqual(fnn.W[2].shape, (neurons_L2, features_Y))

        self.assertEqual(fnn.b[0].shape, (neurons_L1, ))
        self.assertEqual(fnn.b[1].shape, (neurons_L2, ))
        self.assertEqual(fnn.b[2].shape, (features_Y, ))

    def test_fnn_1_1_1(self):
        fnn_layers = [Layer(1, None, None),
                      Layer(1, identity, identity_prime), 
                      Layer(1, identity, identity_prime)]

        fnn = Fnn(w_init = zeros_init, b_init = ones_init, layers = fnn_layers, alg = mse(fnn_layers))

        X = as_column_vector([1.0, 2.0, 3.0, 4.0])
        T = as_column_vector([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(X.shape, T.shape)

        Y, Z, A = fnn.forward(X)

        self.assertEqual(Y.shape, T.shape)
        self.assertTrue(np.allclose(Y, T))
        
        for z, a, layer in zip(Z, A, fnn_layers[1:]):
            self.assertEqual(z.shape, (len(X), layer.neurons))
            self.assertEqual(a.shape, z.shape)
        
        dW, db, _ = fnn.gradients(X, Z, A, T)
        fnn.update_W_b(dW, db, 0.1)

        Y1 = fnn.forward(X).Y
        T1 = as_column_vector([1.0, 1.0, 1.0, 1.0])        

        self.assertTrue(np.allclose(Y1, T1))
        
    def test_fnn_1_3_1(self):
        fnn_layers = [Layer(1, None, None),
                      Layer(3, identity, identity_prime), 
                      Layer(1, identity, identity_prime)]

        fnn = Fnn(w_init = ones_init, b_init = ones_init, layers = fnn_layers, alg = mse(fnn_layers))

        X = as_column_vector([1.0,  2.0,  3.0,  4.0])
        T = as_column_vector([7.0, 10.0, 13.0, 16.0])
        
        Y, Z, A = fnn.forward(X)

        self.assertEqual(Y.shape, T.shape)
        self.assertTrue(np.allclose(Y, T))

        for z, a, layer in zip(Z, A, fnn_layers[1:]):
            self.assertEqual(z.shape, (len(X), layer.neurons))
            self.assertEqual(a.shape, z.shape)

    def test_fnn_1_1_3_bias_broadcast(self):
        fnn_layers = [Layer(1, None, None),
                      Layer(1, identity, identity_prime), 
                      Layer(3, identity, identity_prime)]

        fnn = Fnn(w_init = zeros_init, b_init = arrange_init, layers = fnn_layers, alg = mse(fnn_layers))

        X = as_column_vector([1.0,  2.0,  3.0,  4.0])
        T = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        
        Y, Z, A = fnn.forward(X)

        self.assertEqual(Y.shape, T.shape)
        self.assertTrue(np.allclose(Y, T))

        for z, a, layer in zip(Z, A, fnn_layers[1:]):
            self.assertEqual(z.shape, (len(X), layer.neurons))
            self.assertEqual(a.shape, z.shape)

    def test_fnn_1_2_1_backprop_W(self):
        random.seed(22)
        np.random.seed(22)

        fnn_layers = [Layer(1, None, None),
                      Layer(2, identity, identity_prime), 
                      Layer(1, identity, identity_prime)]
        
        fnn = Fnn(w_init = rand_norm_init, b_init = rand_norm_init, layers = fnn_layers, alg = mse(fnn_layers))

        X = as_column_vector([-0.5,  0.0, 0.5])
        T = as_column_vector([ 0.0, 0.25, 0.5])
        #X = as_column_vector([-0.5,  -0.5, -0.5])
        #T = as_column_vector([ 0.0, 0.0, 0.0])
        #X = as_column_vector([-0.5])
        #T = as_column_vector([ 0.0])
        
        _, Z0, A0 = fnn.forward(X)
        
        epsilon = 0.000001
        W_backup = fnn.W[1][0, 0]
        
        fnn.W[1][0, 0] += epsilon
        Y_plus, *_ = fnn.forward(X)
        fnn.W[1][0, 0] = W_backup

        L_plus = mse_loss_mean(T, Y_plus)

        fnn.W[1][0, 0] -= epsilon
        Y_minus, *_ = fnn.forward(X)
        fnn.W[1][0, 0] = W_backup

        L_minus = mse_loss_mean(T, Y_minus)
        
        g_num = (L_plus - L_minus) / (2 * epsilon)        

        dW, db, _ = fnn.gradients(X, Z0, A0, T)
        fnn.update_W_b(dW, db, 0.1)

        g_bp = dW[1][0,0]
        self.assertTrue(math.isclose(g_bp, g_num))

        abs_err = abs(g_num - g_bp)
        rel_err = abs_err / max(1, abs(g_num), abs(g_bp))
        self.assertTrue(abs_err < 0.00000001)
        self.assertTrue(rel_err < 0.000001)

    def test_fnn_1_2_1_backprop_b(self):
        random.seed(22)
        np.random.seed(22)

        fnn_layers = [Layer(1, None, None),
                      Layer(2, identity, identity_prime), 
                      Layer(1, identity, identity_prime)]
        
        fnn = Fnn(w_init = rand_norm_init, b_init = rand_norm_init, layers = fnn_layers, alg = mse(fnn_layers))

        X = as_column_vector([-0.5,  0.0, 0.5])
        T = as_column_vector([ 0.0, 0.25, 0.5])
        
        _, Z0, A0 = fnn.forward(X)
        
        epsilon = 0.000001
        b_backup = fnn.b[1][0]
        
        fnn.b[1][0] += epsilon
        Y_plus, *_ = fnn.forward(X)
        fnn.b[1][0] = b_backup

        L_plus = mse_loss_mean(T, Y_plus)

        fnn.b[1][0] -= epsilon
        Y_minus, *_ = fnn.forward(X)
        fnn.b[1][0] = b_backup

        L_minus = mse_loss_mean(T, Y_minus)
        
        g_num = (L_plus - L_minus) / (2 * epsilon)        

        dW, db, _ = fnn.gradients(X, Z0, A0, T)
        fnn.update_W_b(dW, db, 0.1)
        
        g_bp = db[1][0]
        self.assertTrue(math.isclose(g_bp, g_num))

        abs_err = abs(g_num - g_bp)
        rel_err = abs_err / max(1, abs(g_num), abs(g_bp))
        self.assertTrue(abs_err < 0.00000001)
        self.assertTrue(rel_err < 0.000001)

    def test_fnn_1_2_1_backprop_indicies_correctness(self):
        random.seed(22)
        np.random.seed(22)

        fnn_layers = [Layer(1, None, None),
                      Layer(2, tanh, tanh_prime), 
                      Layer(1, identity, identity_prime)]
        
        fnn = Fnn(w_init = rand_norm_init, b_init = rand_norm_init, layers = fnn_layers, alg = mse(fnn_layers))

        X = as_column_vector([-0.5,  0.0, 0.5])
        T = as_column_vector([ 0.0, 0.25, 0.5])
        
        _, Z, A = fnn.forward(X)
        
        A.insert(0, X)

        dL_dA_out = (A[2] - T) / len(X)
        dZ_out = dL_dA_out * fnn.layers[2].a_fn_prime(A[2])
       
        G_back = dZ_out @ fnn.W[1].T
        G_gate = fnn.layers[1].a_fn_prime(A[1])

        dZ_hidden_expected = G_back * G_gate
        dW_hidden_expected = A[0].T @ dZ_hidden_expected

        A.pop(0)

        dW, db, dZ = fnn.gradients(X, Z, A, T)
        fnn.update_W_b(dW, db, 0.1)

        self.assertTrue(np.allclose(dW[0], dW_hidden_expected))
        self.assertTrue(np.allclose(dZ[1], dZ_out))        
        self.assertTrue(np.allclose(dZ[0], dZ_hidden_expected))

    def test_fnn_1_2_1_backprop_W_hid(self):
        random.seed(22)
        np.random.seed(22)

        fnn_layers = [Layer(1, None, None),
                      Layer(2, tanh, tanh_prime), 
                      Layer(1, identity, identity_prime)]
        
        fnn = Fnn(w_init = rand_norm_init, b_init = rand_norm_init, layers = fnn_layers, alg = mse(fnn_layers))

        X = as_column_vector([-0.5,  0.0, 0.5])
        T = as_column_vector([ 0.0, 0.25, 0.5])
        
        _, Z0, A0 = fnn.forward(X)
        
        epsilon = 0.000001
        W_backup = fnn.W[0][0, 0]
        
        fnn.W[0][0, 0] += epsilon
        Y_plus, *_ = fnn.forward(X)
        fnn.W[0][0, 0] = W_backup

        L_plus = mse_loss_mean(T, Y_plus)

        fnn.W[0][0, 0] -= epsilon
        Y_minus, *_ = fnn.forward(X)
        fnn.W[0][0, 0] = W_backup

        L_minus = mse_loss_mean(T, Y_minus)
        
        g_num = (L_plus - L_minus) / (2 * epsilon)        

        dW, db, dZ = fnn.gradients(X, Z0, A0, T)
        fnn.update_W_b(dW, db, 0.1)
        
        g_bp = dW[0][0,0]
        self.assertTrue(math.isclose(g_bp, g_num))

        abs_err = abs(g_num - g_bp)
        rel_err = abs_err / max(1, abs(g_num), abs(g_bp))
        self.assertTrue(abs_err < 0.00000001)
        self.assertTrue(rel_err < 0.000001)

    def test_fnn_1_2_1_one_step_SGD_lowers_loss(self):
        random.seed(22)
        np.random.seed(22)

        fnn_layers = [Layer(1, None, None),
                      Layer(2, tanh, tanh_prime), 
                      Layer(1, identity, identity_prime)]
        
        fnn = Fnn(w_init = rand_norm_init, b_init = rand_norm_init, layers = fnn_layers, alg = mse(fnn_layers))

        X = as_column_vector([-0.5,  0.0, 0.5])
        T = as_column_vector([ 0.0, 0.25, 0.5])
        
        Y0, Z0, A0 = fnn.forward(X)
        
        L0 = mse_loss_mean(T, Y0)
        
        dW, db, _ = fnn.gradients(X, Z0, A0, T)

        fnn.update_W_b(dW, db, 0.1)

        Y1, *_ = fnn.forward(X)

        L1 = mse_loss_mean(T, Y1)

        self.assertLess(L1, L0)

    def test_bce_logits_step_lowers_loss(self):
        B = 64
        X = np.random.randn(B, 2)
        T = (np.random.rand(B, 1) > 0.5).astype(float)

        layers = [
            Layer(2, None, None),
            Layer(4, tanh, tanh_prime),
            Layer(1, identity, identity_prime),   # logits head
        ]
        fnn = Fnn(w_init=Xavier_init, b_init=zeros_init, layers=layers, alg = bce())

        _, Z0, A0 = fnn.forward(X)
        L0 = bce_loss_mean(T, Z0[-1])

        dW, db, _ = fnn.gradients(X, Z0, A0, T)
        fnn.update_W_b(dW, db, eta=0.05)

        Y1, _, _ = fnn.forward(X)
        L1 = bce_loss_mean(T, Y1)

        self.assertLess(L1, L0)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFnn)
    unittest.TextTestRunner(verbosity=2).run(suite)