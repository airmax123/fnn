import numpy as np
import matplotlib.pyplot as plt
from fnn import *

def fit_a_sin():
    np.set_printoptions(linewidth=160, precision=2, suppress=True)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[80, 20]})

    fnn_layers = [Layer(1, None, None),
                  Layer(10, LeakyReLU, LeakyReLU_prime),
                  Layer(10, LeakyReLU, LeakyReLU_prime),
                  Layer(1, identity, identity_prime)]

    fnn = Fnn(w_init = He_init, b_init = zeros_init, layers = fnn_layers)
    
    sample_count = 400
    X = np.random.uniform(-math.pi/4, math.pi/4, (sample_count, 1))
    T = 0.3 * np.sin(X / 0.1) + 0.4
    #T += np.random.normal(0, 0.02, (sample_count, 1)) # Add small Gaussian noise to targets

    ax1.scatter(X, T, s = 30, marker = "^")

    log = fnn.train(X, T, 2000, 32, 0.1, eta_decay_rate=0.999)
    
    X = np.linspace(-math.pi/4, math.pi/4, sample_count).reshape(sample_count, 1)
    Y, *_ = fnn.forward(X)
        
    ax1.plot(X, Y, linewidth=0.8, color = 'r')
    # plot train loss
    ax2.scatter(range(log[:, 0].size), log[:, 0], s = 15)
    # plot eval loss
    ax2.scatter(range(log[:, 1].size), log[:, 1], s = 2, color = 'r')

    plt.show()
     
if __name__ == '__main__':
    fit_a_sin()