import numpy as np

class Optimizer:
    def init_state(self, W, B): pass

    def step(self, W, B, dW, dB, learning_rate): pass

class SGD(Optimizer):
    def init_state(self, W, B): return None

    def step(self, W, B, dW, dB, learning_rate):
        for i in range(len(W)):
            W[i] -= learning_rate * dW[i]
            B[i] -= learning_rate * dB[i]
        return W, B

class Momentum(Optimizer):
    def __init__(self, beta=0.9):
        self.beta = beta
        self.vW = None; self.vB = None

    def init_state(self, W, B):
        self.vW = [np.zeros_like(w) for w in W]
        self.vB = [np.zeros_like(b) for b in B]

    def step(self, W, B, dW, dB, learning_rate):
        for i in range(len(W)):
            self.vW[i] = self.beta*self.vW[i] + (1-self.beta)*dW[i]
            self.vB[i] = self.beta*self.vB[i] + (1-self.beta)*dB[i]
            W[i] -= learning_rate * self.vW[i]
            B[i] -= learning_rate * self.vB[i]
        return W, B

class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.mW = self.mB = self.vW = self.vB = None
        self.t = 0

    def init_state(self, W, B):
        self.mW = [np.zeros_like(w) for w in W]
        self.mB = [np.zeros_like(b) for b in B]
        self.vW = [np.zeros_like(w) for w in W]
        self.vB = [np.zeros_like(b) for b in B]
        self.t = 0

    def step(self, W, B, dW, dB, learning_rate):
        self.t += 1
        for i in range(len(W)):
            # moments
            self.mW[i] = self.b1*self.mW[i] + (1-self.b1)*dW[i]
            self.mB[i] = self.b1*self.mB[i] + (1-self.b1)*dB[i]
            self.vW[i] = self.b2*self.vW[i] + (1-self.b2)*(dW[i]**2)
            self.vB[i] = self.b2*self.vB[i] + (1-self.b2)*(dB[i]**2)
            # bias correction
            mW_hat = self.mW[i] / (1 - self.b1**self.t)
            mB_hat = self.mB[i] / (1 - self.b1**self.t)
            vW_hat = self.vW[i] / (1 - self.b2**self.t)
            vB_hat = self.vB[i] / (1 - self.b2**self.t)
            # update
            W[i] -= learning_rate * mW_hat / (np.sqrt(vW_hat) + self.eps)
            B[i] -= learning_rate * mB_hat / (np.sqrt(vB_hat) + self.eps)
        return W, B
