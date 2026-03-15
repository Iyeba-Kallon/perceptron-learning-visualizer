import numpy as np

class Optimizer:
    """
    Base class for all Optimizers.
    Optimizers update model parameters (weights) based on calculated gradients.
    """
    def update(self, w, grad):
        """
        Update the weights.
        
        Args:
            w (ndarray): Current weights.
            grad (ndarray): Gradient of the loss with respect to weights.
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) Optimizer.
    The most basic optimization algorithm that updates weights by taking 
    steps proportional to the negative of the gradient.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, w, grad):
        # Weight update rule: w = w - eta * grad
        return w - self.learning_rate * grad

class Momentum(Optimizer):
    """
    Gradient Descent with Momentum.
    Accelerates SGD in the relevant direction and dampens oscillations by 
    adding a fraction of the previous update to the current update.
    """
    def __init__(self, learning_rate=0.01, beta=0.9):
        """
        Args:
            learning_rate (float): Step size for updates.
            beta (float): Momentum coefficient (typically 0.9).
        """
        self.learning_rate = learning_rate
        self.beta = beta
        self.v = None  # Velocity vector

    def update(self, w, grad):
        # Initialize velocity vector on first call
        if self.v is None:
            self.v = np.zeros_like(w)
        
        # Exponentially weighted moving average of the gradient
        self.v = self.beta * self.v + (1 - self.beta) * grad
        
        # Update weights using the velocity
        return w - self.learning_rate * self.v
