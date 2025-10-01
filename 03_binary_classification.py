import numpy as np
import matplotlib.pyplot as plt
from fnn import *

def estimate_pos_weight(T):
    n_pos = np.sum(T)
    n_neg = len(T) - n_pos
    eps = 1e-12
    return float(n_neg / (n_pos + eps))    

def print_classification_metrics(T_eval, Y_eval):
    P_eval = sigmoid(Y_eval)
    prediction = (P_eval >= 0.5).astype(int)
    accuracy  = np.mean(prediction == T_eval)
    precision = np.sum((prediction == 1) & (T_eval == 1)) / max(np.sum(prediction == 1), 1)
    recall  = np.sum((prediction == 1) & (T_eval == 1)) / max(np.sum(T_eval == 1), 1)
    print(f"Eval acc={accuracy:.3f}  precision={precision:.3f}  recall={recall:.3f}")

def binary_classify():
    np.set_printoptions(linewidth=160, precision=2, suppress=True)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[80, 20]})

    sample_count = 1000
    X, T = make_circles(sample_count, imbalance = 0.05)

    fnn_layers = [Layer(2, None, None),
                  Layer(4, tanh, tanh_prime),
                  Layer(4, tanh, tanh_prime),
                  Layer(1, identity, identity_prime)] # prime function will be unused in case of BCE
    
    fnn = Fnn(w_init = Xavier_init, b_init = zeros_init, layers = fnn_layers, 
        alg = bce_weighted(pos_weight = estimate_pos_weight(T), neg_weight = 1.0))
    
    ax1.set_aspect('equal', 'box')
    ax1.scatter(X[:,0], X[:,1], c=T.ravel(), s=8)

    X_train, X_eval, T_train, T_eval = train_test_split(X, T)
    log = fnn.train(X_train, X_eval, T_train, T_eval, 1000, 64, 0.05, eta_decay_rate=0.999)

    print_classification_metrics(T_eval, fnn.forward(X_eval).Y)

    xs = np.linspace(-2, 2, 200)
    ys = np.linspace(-2, 2, 200)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)
    logits, *_ = fnn.forward(grid)
    class_1_probability = sigmoid(logits)
    ZZ = (class_1_probability.reshape(XX.shape) >= 0.5).astype(int)
    
    ax1.contourf(XX, YY, ZZ, alpha=0.2)
        
    # plot train loss
    ax2.scatter(range(log[:, 0].size), log[:, 0], s = 15)
    # plot eval loss
    ax2.scatter(range(log[:, 1].size), log[:, 1], s = 2, color = 'r')

    plt.show()
     
if __name__ == '__main__':
    binary_classify()