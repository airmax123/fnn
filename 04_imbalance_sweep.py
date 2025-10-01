from decimal import Overflow
import numpy as np
import matplotlib.pyplot as plt

from fnn import *

# -----------------------
# Data: imbalanced circles
# -----------------------
def make_circles_balanced(n=4000, r_inner=0.7, r_outer=1.3, noise=0.08):
    n0 = n // 2
    n1 = n - n0

    th0 = np.random.uniform(0, 2*np.pi, n0)
    r0  = r_inner + np.random.normal(0, noise, n0)
    X0  = np.stack([r0*np.cos(th0), r0*np.sin(th0)], axis=1)
    T0  = np.zeros((n0, 1))

    th1 = np.random.uniform(0, 2*np.pi, n1)
    r1  = r_outer + np.random.normal(0, noise, n1)
    X1  = np.stack([r1*np.cos(th1), r1*np.sin(th1)], axis=1)
    T1  = np.ones((n1, 1))

    X = np.vstack([X0, X1])
    T = np.vstack([T0, T1])
    idx = np.random.permutation(len(X))
    return X[idx], T[idx]

def make_circles_imbalanced(n=4000, pos_frac=0.05, **kwargs):
    """Create dataset with approx pos_frac positives (class=1)."""
    X, T = make_circles_balanced(n=n, **kwargs)

    pos_idx = np.where(T.ravel() == 1)[0]
    neg_idx = np.where(T.ravel() == 0)[0]

    n_pos_target = max(1, int(round(pos_frac * len(T))))
    # If too many positives, downsample; if too few, downsample negatives
    if len(pos_idx) >= n_pos_target:
        pos_keep = np.random.choice(pos_idx, size=n_pos_target, replace=False)
        X_pos, T_pos = X[pos_keep], T[pos_keep]
        X_neg, T_neg = X[neg_idx], T[neg_idx]  # keep all negatives
    else:
        # If very small pos_frac, this branch rarely triggers with balanced base,
        # but keep it robust.
        X_pos, T_pos = X[pos_idx], T[pos_idx]
        n_neg_target = len(T) - len(X_pos)
        neg_keep = np.random.choice(neg_idx, size=n_neg_target, replace=False)
        X_neg, T_neg = X[neg_keep], T[neg_keep]

    X_new = np.vstack([X_neg, X_pos])
    T_new = np.vstack([T_neg, T_pos])

    idx = np.random.permutation(len(X_new))
    return X_new[idx], T_new[idx]

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
        X, T = make_circles_imbalanced(n=4000, pos_frac=pos_frac)
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
