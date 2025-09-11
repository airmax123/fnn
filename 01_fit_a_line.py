import numpy as np
import matplotlib.pyplot as plt
from fnn import *

def fit_a_line():
    np.set_printoptions(linewidth=160, precision=2, suppress=True)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[80, 20]})

    fnn_layers = [Layer(1, None, None),
                  Layer(1, identity, identity_prime)]

    fnn = Fnn(w_init = He_init, b_init = zeros_init, layers = fnn_layers)
    
    sample_count = 200
    X = np.random.uniform(-2, 2, (sample_count, 1))
    T = 3*X + 4 
    T += np.random.normal(0, 0.05, (sample_count, 1)) # Add small Gaussian noise to targets

    ax1.scatter(X, T, s = 30, marker = "^")

    X_train, X_eval, T_train, T_eval = train_test_split(X, T)

    log = fnn.train(X_train, X_eval, T_train, T_eval, 2000, 10, 0.1)
    
    X = np.linspace(-2, 2, sample_count).reshape(sample_count, 1)
    Y, *_ = fnn.forward(X)
        
    ax1.plot(X, Y, linewidth=0.8, color = 'r')
    # plot train loss
    ax2.scatter(range(log[:, 0].size), log[:, 0], s = 15)
    # plot eval loss
    ax2.scatter(range(log[:, 1].size), log[:, 1], s = 2, color = 'r')
    # plot patience
    #ax2.scatter(range(log[:, 2].size), log[:, 2], s = 2, marker = "^", color = 'y')

    plt.show()
     
if __name__ == '__main__':
    fit_a_line()