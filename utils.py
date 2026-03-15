"""
Numerical utilities and activation functions for Neural Networks.
"""
import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function: 1 / (1 + exp(-z)).
    Includes clipping to prevent overflow (Numerical Stability).
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    s = sigmoid(z)
    return s * (1 - s)

def hardlim(z):
    """Hard limit activation function: Returns 1 if z >= 0, else 0."""
    return np.where(z >= 0, 1.0, 0.0)

def hardlim_prime(z):
    """Derivative of hardlim. Zero almost everywhere."""
    return np.zeros_like(z)

def hardlims(z):
    """Symmetric hard limit: Returns 1 if z >= 0, else -1."""
    return np.where(z >= 0, 1.0, -1.0)

def hardlims_prime(z):
    """Derivative of symmetric hard limit. Zero almost everywhere."""
    return np.zeros_like(z)

def relu(z):
    """Rectified Linear Unit (ReLU): Returns max(0, z)."""
    return np.maximum(0, z)

def relu_prime(z):
    """Derivative of ReLU: Returns 1 if z > 0, else 0."""
    return (z > 0).astype(float)

def tanh(z):
    """Hyperbolic tangent activation function."""
    return np.tanh(z)

def tanh_prime(z):
    """Derivative of tanh."""
    return 1.0 - np.tanh(z)**2

def identity(z):
    """Identity function (Linear activation) used primarily in Adaline."""
    return z

def identity_prime(z):
    """Derivative of the identity function."""
    return np.ones_like(z)

def softmax(z):
    """
    Softmax function for multi-class distributions.
    Subtracts max value for numerical stability.
    """
    # handle 2D input (samples, features)
    exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.
    Support both binary (0/1) and categorical (one-hot) formats.
    """
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    else:
        # Binary threshold comparison
        return np.mean(y_true == (y_pred >= 0.5))

def mse(y_true, y_pred):
    """Mean Squared Error: Average of squared differences."""
    return np.mean((y_true - y_pred)**2)

def cross_entropy_loss(y_true, y_pred):
    """Cross-Entropy Loss for categorical probability distributions."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def logic_gate_data(gate_type):
    """
    Returns features (X) and labels (y) for standard logic gates.
    
    Args:
        gate_type (str): 'AND', 'OR', 'NAND', or 'XOR'.
        
    Returns:
        tuple: (X, y)
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    gates = {
        'AND':  np.array([0, 0, 0, 1]),
        'OR':   np.array([0, 1, 1, 1]),
        'NAND': np.array([1, 1, 1, 0]),
        'XOR':  np.array([0, 1, 1, 0])
    }
    y = gates.get(gate_type.upper(), gates['AND'])
    return X, y
