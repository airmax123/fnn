import numpy as np
import matplotlib.pyplot as plt

# Simulate a noisy gradient signal
np.random.seed(0)
steps = 200
true_grad = np.linspace(1.0, 0.0, steps)      # gradually decaying slope
noise = np.random.normal(0, 0.3, steps)        # gradient noise
g = true_grad + noise

# Adam hyperparameters
beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 0.01

# Initialize
m = np.zeros(steps)
v = np.zeros(steps)
m_hat = np.zeros(steps)
v_hat = np.zeros(steps)
theta = np.zeros(steps)
eff_lr = np.zeros(steps)

for t in range(1, steps):
    # Update biased moment estimates
    m[t] = beta1 * m[t-1] + (1 - beta1) * g[t]
    v[t] = beta2 * v[t-1] + (1 - beta2) * (g[t] ** 2)

    # Bias correction
    m_hat[t] = m[t] / (1 - beta1 ** t)
    v_hat[t] = v[t] / (1 - beta2 ** t)

    # Effective learning rate (per-parameter)
    eff_lr[t] = lr / (np.sqrt(v_hat[t]) + eps)

    # Parameter update
    theta[t] = theta[t-1] - eff_lr[t] * m_hat[t]

# ---- Plot ----
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(g, label="Raw gradients", alpha=0.6)
plt.plot(m_hat, label="1st moment (m̂)", linewidth=2)
plt.legend(); plt.ylabel("Gradient / Momentum")

plt.subplot(3, 1, 2)
plt.plot(np.sqrt(v_hat), label="√v̂ (RMS grad)", color="tab:orange")
plt.legend(); plt.ylabel("Magnitude")

plt.subplot(3, 1, 3)
plt.plot(eff_lr * 1000, label="Effective LR ×1000", color="tab:green")
plt.legend(); plt.ylabel("Scaled LR")
plt.xlabel("Step")

plt.tight_layout()
plt.show()
