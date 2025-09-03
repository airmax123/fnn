import numpy as np
import matplotlib.pyplot as plt
from fnn import *

def fit_a_line():
    np.set_printoptions(linewidth=160, precision=2, suppress=True)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[80, 20]})

    rng = np.random.default_rng(42)

    fnn_layers = [Layer(1, None, None),
                  #Layer(32, tanh, tanh_prime),
                  Layer(32, tanh, tanh_prime),
                  Layer(1, identity, identity_prime)]

    fnn = Fnn(w_init = Xavier_init, b_init = Xavier_init, layers = fnn_layers)
    
    sample_count = 300
    X = np.random.uniform(-math.pi/4, math.pi/4, (sample_count, 1))
    T = 0.3 * np.sin(X / 0.1) + 0.4
    T += np.random.normal(0, 0.05, (sample_count, 1)) # Add small Gaussian noise to targets

    ax1.scatter(X, T, s = 30, marker = "^")

    log = fnn.train(X, T, 2000, 240, 0.1, target_loss_diff = 0.000001)
    
    X = np.random.uniform(-math.pi/4, math.pi/4, (sample_count, 1))
    Y, *_ = fnn.forward(X)
        
    ax1.scatter(X, Y, s = 10, marker = "*", color = 'r')
    # plot train loss
    ax2.scatter(range(log[:, 0].size), log[:, 0], s = 10)
    # plot eval loss
    ax2.scatter(range(log[:, 1].size), log[:, 1], s = 2, color = 'r')

    plt.show()
     
if __name__ == '__main__':
    fit_a_line()