import numpy as np
import matplotlib.pyplot as plt

from fnn import *

plt.ion()

opts = {
    "SGD": SGD(),
    "Momentum": Momentum(beta=0.9),
    "Adam": Adam()
}

lrs = {
    "SGD":0.02,
    "Momentum":0.02,
    "Adam":0.002
}

if __name__ == "__main__":
    pos_frac = 0.05

    X, T = make_circles(n=1000, r_inner=0.7, r_outer=1.0, imbalance=pos_frac)
    X_train, X_eval, T_train, T_eval = train_test_split(X, T, train_size=0.8)

    n_pos = np.sum(T_train, dtype=int)
    n_neg = len(T_train) - n_pos
    auto_pw = float(n_neg / (n_pos + 1e-12))

    fig, axes = plt.subplots(nrows=len(opts), sharex=True)
    fig.tight_layout()
    axe_index = 0
    for name, opt in opts.items():
        layers = [
            Layer(2, None, None),
            Layer(8, LeakyReLU, LeakyReLU_prime),
            Layer(8, LeakyReLU, LeakyReLU_prime),
            Layer(1, identity, identity_prime),  # logits head for BCE
        ]

        fnn = Fnn(w_init=He_init, b_init=zeros_init, layers=layers,
                  alg=bce_weighted(pos_weight=auto_pw), optmizer=opt)

        log = fnn.train(X_train, X_eval, T_train, T_eval,
            max_epochs=2000, batch_size=64, eta=lrs[name], eta_decay_rate=0.999,
            patience=80, rel_tol=3e-3)

        # plot train loss
        axes[axe_index].set_title(name)
        axes[axe_index].scatter(range(log[:, 0].size), log[:, 0], s = 15)
        # plot eval loss
        axes[axe_index].scatter(range(log[:, 1].size), log[:, 1], s = 2, color = 'r')

        axe_index += 1
        plt.show()
    
    plt.pause(3600)