from decimal import Overflow
import numpy as np
import matplotlib.pyplot as plt

from fnn import *

def make_circles(n=1000, r_inner=0.7, r_outer=1.3, noise=0.08, imbalance=0, seed=None):
    rng = np.random.default_rng(seed)

    if imbalance > 0:
        # imbalance = fraction of positives
        n1 = int(n * imbalance)   # positives
        n0 = n - n1               # negatives
    else:
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

# -----------------------
# Metrics
# -----------------------
def precision_recall_at_thresholds(T, P, thresholds):
    """T: (B,1) {0,1}, P: (B,1) probabilities, thresholds: (K,)"""
    T = T.astype(int).ravel()
    P = P.ravel()
    prec, rec = [], []
    for thr in thresholds:
        pred = (P >= thr).astype(int)
        tp = np.sum((pred == 1) & (T == 1))
        fp = np.sum((pred == 1) & (T == 0))
        fn = np.sum((pred == 0) & (T == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec.append(precision)
        rec.append(recall)
    return np.array(prec), np.array(rec)

# -----------------------
# Training helper
# -----------------------
def train_and_eval(X_train, X_eval, T_train, T_eval, pos_frac, pos_weight_values):
    results = []  # list of dicts per pos_weight

    for pos_weight in pos_weight_values:
        print(f"[pos_frac={pos_frac:.2%}] Train positives: {int(n_pos)}/{len(T_train)} auto pos_weight ≈ {pos_weight:.2f}")

        # Model
        layers = [
            Layer(2, None, None),
            Layer(8, LeakyReLU, LeakyReLU_prime),
            Layer(8, LeakyReLU, LeakyReLU_prime),
            Layer(1, identity, identity_prime),  # logits head for BCE
        ]
        fnn = Fnn(w_init=He_init, b_init=zeros_init, layers=layers, alg=bce_weighted(pos_weight=pos_weight))

        fnn.train(X_train, X_eval, T_train, T_eval,
            max_epochs=3000, batch_size=64, eta=0.02,
            eta_decay_rate=0.999, patience=80)

        # Eval probs
        logits_eval, *_ = fnn.forward(X_eval)
        P_eval = sigmoid(logits_eval)

        # PR vs thresholds
        thresholds = np.linspace(0.02, 0.98, 49)
        prec, rec = precision_recall_at_thresholds(T_eval, P_eval, thresholds)

        # Also report accuracy at 0.5 for reference
        acc = np.mean((P_eval >= 0.5).astype(int) == T_eval)

        results.append({
            "pos_weight": pos_weight,
            "thresholds": thresholds,
            "precision": prec,
            "recall": rec,
            "acc@0.5": float(acc),
        })

    return results

# -----------------------
# Main sweep
# -----------------------
if __name__ == "__main__":
    imbalances = [0.01, 0.05, 0.20] # 1%, 5%, 20% positives

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, pos_frac in zip(axes, imbalances):
        X, T = make_circles(n=1000, imbalance=pos_frac)
        X_train, X_eval, T_train, T_eval = train_test_split(X, T, train_size=0.8)

        n_pos = np.sum(T_train)
        n_neg = len(T_train) - n_pos
        auto_pw = float(n_neg / (n_pos + 1e-12))

        pos_weight_values = [1.0, auto_pw, 2.0 * auto_pw]  # baseline, balanced, aggressive

        results = train_and_eval(X_train, X_eval, T_train, T_eval, pos_frac, pos_weight_values)

        # Plot precision/recall vs threshold curves
        for res in results:
            thr = res["thresholds"]
            ax.plot(thr, res["precision"], label=f"pw={res['pos_weight']:.2f} | precision")
            ax.plot(thr, res["recall"],    linestyle="--", label=f"pw={res['pos_weight']:.2f} | recall")

        ax.set_title(f"Positives = {int(pos_frac*100)}%")
        ax.set_xlabel("Threshold")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Precision / Recall")
    axes[0].legend(fontsize=8, loc="lower left", ncol=1)
    plt.tight_layout()
    plt.show()
