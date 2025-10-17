from decimal import Overflow
import numpy as np
import matplotlib.pyplot as plt

from fnn import *

plt.ion()

def precision_recall_at_thresholds(T, P, thresholds):
    T = T.ravel()
    P = P.ravel()

    prec, rec, f1 = [], [], []

    for thr in thresholds:
        pred = (P >= thr).astype(int)
        tp = np.sum((pred == 1) & (T == 1))
        fp = np.sum((pred == 1) & (T == 0))
        fn = np.sum((pred == 0) & (T == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score  = 2 * precision * recall / (precision + recall + 1e-12)

        prec.append(precision)
        rec.append(recall)
        f1.append(f1_score)

    return np.array(prec), np.array(rec), np.array(f1)

def plot_probability_histograms(pos_frac, results, T_eval):
    fig, axes = plt.subplots(nrows=2, ncols=len(results), sharey='row', gridspec_kw={'height_ratios':[50, 50]})

    for col, res in zip(range(0, len(results)), results):
        axes[0, col].set_box_aspect(1)

        pos_weight = res["pos_weight"]
        thresholds = res.get("thresholds")
        f1  = res.get("f1")
        best_idx = np.argmax(f1)
        best_thr = thresholds[best_idx] if thresholds is not None else None

        P_eval_flat = res["P_eval"].ravel()
        T_eval_flat = T_eval.ravel()

        pos_scores = P_eval_flat[T_eval_flat == 1]
        neg_scores = P_eval_flat[T_eval_flat == 0]

        bins = np.linspace(0, 1, 50)
        axes[0, col].hist(neg_scores, bins=bins, alpha=0.6, label="negatives", color="tab:blue", density=True)
        axes[0, col].hist(pos_scores, bins=bins, alpha=0.6, label="positives", color="tab:orange", density=True)

        if best_thr is not None:
            axes[0, col].axvline(best_thr, color="k", linestyle="--", linewidth=1.5, label=f"best F1 thr={best_thr:.2f}")

        axes[0, col].set_title(f"pos_weight={pos_weight:.1f}")
        axes[0, col].set_xlabel("Predicted probability (sigmoid(logit))")
        axes[0, col].set_xlim(0, 1)
        axes[0, col].grid(alpha=0.3)
        axes[0, col].legend(fontsize=8, loc="lower left", ncol=1)

        axes[1, col].set_box_aspect(1)

        axes[1, col].plot(res["thresholds"], res["precision"], label=f"pw={res['pos_weight']:.2f} | precision")
        axes[1, col].plot(res["thresholds"], res["recall"],    linestyle="--", label=f"pw={res['pos_weight']:.2f} | recall")

        axes[1, col].set_xlabel("Threshold")
        axes[1, col].grid(alpha=0.3)
        axes[1, col].legend(fontsize=8, loc="lower left", ncol=1)

    axes[0, 0].set_ylabel("Density")
    
    axes[1, 0].set_ylabel("Precision / Recall")

    fig.suptitle(f"Predicted probability distributions – positives={pos_frac:.1%}")
    
    plt.tight_layout()
    plt.show()

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

        # Forward pass on eval data
        logits_eval, *_ = fnn.forward(X_eval)
        P_eval = sigmoid(logits_eval)
        
        # Compute metrics for threshold sweep
        thresholds = np.linspace(0.02, 0.98, 49)
        prec, rec, f1 = precision_recall_at_thresholds(T_eval, P_eval, thresholds)

        results.append({"pos_weight": pos_weight, "thresholds": thresholds, "precision": prec, "recall": rec, "f1": f1, "P_eval" : P_eval.copy()})
    
    return results

if __name__ == "__main__":
    imbalances = [0.01, 0.05, 0.20] # 1%, 5%, 20% positives

    for pos_frac in imbalances:
        X, T = make_circles(n=1000, r_inner=0.7, r_outer=1.0, imbalance=pos_frac)
        X_train, X_eval, T_train, T_eval = train_test_split(X, T, train_size=0.8)

        n_pos = np.sum(T_train)
        n_neg = len(T_train) - n_pos
        auto_pw = float(n_neg / (n_pos + 1e-12))

        pos_weight_values = [1.0, auto_pw, 2.0 * auto_pw]  # baseline, balanced, aggressive

        results = train_and_eval(X_train, X_eval, T_train, T_eval, pos_frac, pos_weight_values)
        
        plot_probability_histograms(pos_frac, results, T_eval)

    plt.pause(3600)