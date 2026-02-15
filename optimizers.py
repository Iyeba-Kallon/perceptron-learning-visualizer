import numpy as np

class Optimizer:
    def update(self, w, grad):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, w, grad):
        return w - self.learning_rate * grad

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.v = None

    def update(self, w, grad):
        if self.v is None:
            self.v = np.zeros_like(w)
        
        self.v = self.beta * self.v + (1 - self.beta) * grad
        return w - self.learning_rate * self.v
