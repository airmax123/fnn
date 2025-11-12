import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
steps = 300

# --- Noisy 1D gradient stream
true_grad = np.linspace(1.0, 0.0, steps)
noise = np.random.normal(0, 0.35, steps)
g = true_grad + noise

# --- Hyperparams
lr_sgd = 0.03
lr_mom = 0.03; beta = 0.9
lr_adam = 0.01; b1 = 0.9; b2 = 0.999; eps = 1e-8

# --- Traces
theta_sgd = np.zeros(steps)
theta_mom = np.zeros(steps)
theta_adam = np.zeros(steps)

# step magnitudes we’ll plot (abs update per step)
step_sgd = np.zeros(steps)
step_mom = np.zeros(steps)
step_adam = np.zeros(steps)

# Momentum state
v = 0.0  # velocity

# Adam state
m = 0.0; v2 = 0.0
m_hat_tr = np.zeros(steps)
rms_tr = np.zeros(steps)
eff_lr_adam = np.zeros(steps)

for t in range(1, steps):
    # ----- SGD
    delta_sgd = lr_sgd * g[t]                 # actual step
    theta_sgd[t] = theta_sgd[t-1] - delta_sgd
    step_sgd[t] = abs(delta_sgd)

    # ----- Momentum (Polyak)
    v = beta * v + (1 - beta) * g[t]          # momentum buffer (smoothed grad)
    delta_mom = lr_mom * v                     # actual step
    theta_mom[t] = theta_mom[t-1] - delta_mom
    step_mom[t] = abs(delta_mom)

    # ----- Adam
    m = b1 * m + (1 - b1) * g[t]
    v2 = b2 * v2 + (1 - b2) * (g[t]**2)
    m_hat = m / (1 - b1**t)
    v_hat = v2 / (1 - b2**t)
    eff_lr = lr_adam / (np.sqrt(v_hat) + eps)
    delta_adam = eff_lr * m_hat                # actual step
    theta_adam[t] = theta_adam[t-1] - delta_adam

    # logs for plots
    eff_lr_adam[t] = eff_lr
    m_hat_tr[t] = m_hat
    rms_tr[t] = np.sqrt(v_hat)
    step_adam[t] = abs(delta_adam)

# ---- PLOTS ----
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# 1) Direction smoothing (Adam's m̂ vs raw grad)
ax[0].plot(g, color="gray", alpha=0.4, label="raw grad")
ax[0].plot(np.convolve(g, np.ones(10)/10, mode="same"), color="k", alpha=0.5, label="moving avg (ref)")
ax[0].plot(m_hat_tr, color="tab:green", label="Adam 1st moment (m̂)")
ax[0].set_ylabel("Direction")
ax[0].legend(loc="upper right")
ax[0].grid(alpha=0.3)

# 2) Absolute step sizes (what each method actually applies)
ax[1].plot(step_sgd, color="tab:blue", label="|SGD step|")
ax[1].plot(step_mom, color="tab:orange", label="|Momentum step|")
ax[1].plot(step_adam, color="tab:green", label="|Adam step|")
ax[1].set_ylabel("|Δθ| per step")
ax[1].legend(loc="upper right")
ax[1].grid(alpha=0.3)

# 3) Parameter trajectories
ax[2].plot(theta_sgd, color="tab:blue", label="SGD")
ax[2].plot(theta_mom, color="tab:orange", label="Momentum (β=0.9)")
ax[2].plot(theta_adam, color="tab:green", label="Adam")
ax[2].set_xlabel("Step")
ax[2].set_ylabel("θ (position)")
ax[2].legend(loc="best")
ax[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
