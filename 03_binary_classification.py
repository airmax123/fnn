import numpy as np
import matplotlib.pyplot as plt
from fnn import *

def make_circles(n=1000, r_inner=0.7, r_outer=1.3, noise=0.08, seed=0):
    rng = np.random.default_rng(seed)

    n0 = n // 2
    n1 = n - n0

    # inner class (label 0)
    angles0 = rng.uniform(0, 2*np.pi, n0)
    r0 = r_inner + rng.normal(0, noise, n0)
    x0 = np.stack([r0*np.cos(angles0), r0*np.sin(angles0)], axis=1)
    t0 = np.zeros((n0, 1))

    # outer class (label 1)
    angles1 = rng.uniform(0, 2*np.pi, n1)
    r1 = r_outer + rng.normal(0, noise, n1)
    x1 = np.stack([r1*np.cos(angles1), r1*np.sin(angles1)], axis=1)
    t1 = np.ones((n1, 1))

    X = np.vstack([x0, x1])
    T = np.vstack([t0, t1])

    # shuffle
    idx = rng.permutation(len(X))
    return X[idx], T[idx]

def binary_classify():
    np.set_printoptions(linewidth=160, precision=2, suppress=True)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[80, 20]})

    fnn_layers = [Layer(2, None, None),
                  Layer(10, LeakyReLU, LeakyReLU_prime),
                  Layer(10, LeakyReLU, LeakyReLU_prime),
                  Layer(1, sigmoid, sigmoid_prime)]

    fnn = Fnn(w_init = Xavier_init, b_init = zeros_init, layers = fnn_layers)
    
    sample_count = 1000

    X, T = make_circles(sample_count)
    ax1.scatter(X[:,0], X[:,1], c=T.ravel(), s=8)

    X_train, X_eval, T_train, T_eval = train_test_split(X, T)
    log = fnn.train(X_train, X_eval, T_train, T_eval, 2000, 64, 0.05, eta_decay_rate=0.999)

    xs = np.linspace(-2, 2, 200)
    ys = np.linspace(-2, 2, 200)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)
    class_1_probability, *_ = fnn.forward(grid)
    ZZ = (class_1_probability.reshape(XX.shape) >= 0.5).astype(int)
    
    ax1.contourf(XX, YY, ZZ, alpha=0.2)
        
    # plot train loss
    ax2.scatter(range(log[:, 0].size), log[:, 0], s = 15)
    # plot eval loss
    ax2.scatter(range(log[:, 1].size), log[:, 1], s = 2, color = 'r')

    plt.show()
     
if __name__ == '__main__':
    binary_classify()