import numpy as np

def sigmoid(z):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    """ReLU activation function."""
    return np.maximum(0, z)

def relu_prime(z):
    """Derivative of ReLU."""
    return (z > 0).astype(float)

def tanh(z):
    """Hyperbolic tangent activation function."""
    return np.tanh(z)

def tanh_prime(z):
    """Derivative of tanh."""
    return 1.0 - np.tanh(z)**2

def identity(z):
    """Identity activation function (for Adaline)."""
    return z

def identity_prime(z):
    """Derivative of identity."""
    return np.ones_like(z)

def softmax(z):
    """Softmax activation function."""
    exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def accuracy(y_true, y_pred):
    """Calculate accuracy percentage."""
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        # One-hot encoded
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    else:
        # Binary
        return np.mean(y_true == (y_pred >= 0.5))

def mse(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((y_true - y_pred)**2)

def cross_entropy_loss(y_true, y_pred):
    """Cross Entropy Loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def logic_gate_data(gate_type):
    """Returns X, y for logic gates."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    if gate_type == 'AND':
        y = np.array([0, 0, 0, 1])
    elif gate_type == 'OR':
        y = np.array([0, 1, 1, 1])
    elif gate_type == 'NAND':
        y = np.array([1, 1, 1, 0])
    elif gate_type == 'XOR':
        y = np.array([0, 1, 1, 0])
    return X, y
