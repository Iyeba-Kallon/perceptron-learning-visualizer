import numpy as np
from utils import (sigmoid, sigmoid_prime, tanh, tanh_prime, 
                   relu, relu_prime, softmax, hardlim, 
                   hardlim_prime, hardlims, hardlims_prime)

class MLP:
    """
    Multi-Layer Perceptron (MLP) with Backpropagation.
    
    This implementation supports an arbitrary number of hidden layers and 
    various activation functions. It uses the Gradient Descent algorithm 
    to optimize weights based on the Mean Squared Error (MSE).
    """
    def __init__(self, layers, activation='sigmoid', eta=0.01, epochs=100, random_state=1):
        """
        Initialize the Multi-Layer Perceptron.

        Args:
            layers (list): Number of neurons in each layer, e.g., [2, 4, 1].
            activation (str): Activation function for hidden layers ('sigmoid', 'tanh', 'relu', etc.).
            eta (float): Learning rate.
            epochs (int): Number of training iterations.
            random_state (int): Random seed for weight initialization.
        """
        self.layers = layers
        self.activation_name = activation.lower()
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        
        # Internal state for weights, biases, and training data
        self.weights = []
        self.biases = []
        self._init_weights()
        
        self.history = {'loss': [], 'accuracy': []}
        
    def _init_weights(self):
        """
        Initialize weights and biases for each layer.
        Uses specialized initialization (e.g., He initialization for ReLU) 
        to improve convergence.
        """
        rng = np.random.RandomState(self.random_state)
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layers) - 1):
            # Standard Deviation selection based on activation type
            if self.activation_name == 'relu':
                # He Initialization (recommended for ReLU)
                std = np.sqrt(2.0 / self.layers[i])
            else:
                # Xavier/Glorot Initialization (recommended for Sigmoid/Tanh)
                std = np.sqrt(1.0 / self.layers[i])
                
            w = rng.normal(0.0, std, (self.layers[i], self.layers[i+1]))
            b = np.zeros((1, self.layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def _apply_activation(self, z):
        """Apply the chosen activation function to a layer's net input."""
        if self.activation_name == 'sigmoid':
            return sigmoid(z)
        elif self.activation_name == 'tanh':
            return tanh(z)
        elif self.activation_name == 'relu':
            return relu(z)
        elif self.activation_name == 'hardlim':
            return hardlim(z)
        elif self.activation_name == 'hardlims':
            return hardlims(z)
        return z

    def _apply_activation_prime(self, z):
        """Apply the derivative of the chosen activation function."""
        if self.activation_name == 'sigmoid':
            return sigmoid_prime(z)
        elif self.activation_name == 'tanh':
            return tanh_prime(z)
        elif self.activation_name == 'relu':
            return relu_prime(z)
        elif self.activation_name == 'hardlim':
            return hardlim_prime(z)
        elif self.activation_name == 'hardlims':
            return hardlims_prime(z)
        return np.ones_like(z)

    def forward(self, X):
        """
        Perform a forward pass through the network.

        Args:
            X (ndarray): Input data.

        Returns:
            ndarray: The network's final output (activations of the last layer).
        """
        self.a = [X]  # Store activations of each layer (a[0] is the input)
        self.z = []   # Store net inputs (z = Wa + b) of each layer

        for i in range(len(self.weights)):
            z_i = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.z.append(z_i)
            
            # Use Sigmoid for the final output layer (binary classification assumption)
            if i == len(self.weights) - 1:
                a_i = sigmoid(z_i) 
            else:
                a_i = self._apply_activation(z_i)
            self.a.append(a_i)
        
        return self.a[-1]

    def backward(self, X, y):
        """
        Perform the backward pass (Backpropagation) to calculate gradients.

        Args:
            X (ndarray): Input data.
            y (ndarray): Target labels.

        Returns:
            tuple: (grad_weights, grad_biases)
        """
        m = X.shape[0]
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        # 1. Output Layer Error (Delta)
        # Using MSE Loss: Loss = 0.5 * (y - a)^2
        # Delta_L = (a_L - y) * sigma'(z_L)
        target = y.reshape(-1, 1)
        delta = (self.a[-1] - target) * sigmoid_prime(self.z[-1])
        
        grad_w[-1] = np.dot(self.a[-2].T, delta)
        grad_b[-1] = np.sum(delta, axis=0, keepdims=True)

        # 2. Hidden Layer Errors (Backpropagate)
        # Delta_i = (W_i+1.T * Delta_i+1) * sigma'(z_i)
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[i+1].T) * self._apply_activation_prime(self.z[i])
            grad_w[i] = np.dot(self.a[i].T, delta)
            grad_b[i] = np.sum(delta, axis=0, keepdims=True)
            
        return grad_w, grad_b

    def train_step(self, X, y):
        """
        Execute a single iteration of training (Forward + Backward + Update).

        Used by the GUI visualizer to step through training epochs.
        """
        # 1. Forward pass
        output = self.forward(X)
        
        # 2. Backward pass
        grad_w, grad_b = self.backward(X, y)
        
        # 3. Parameter Update (Gradient Descent)
        for i in range(len(self.weights)):
            self.weights[i] -= self.eta * grad_w[i]
            self.biases[i] -= self.eta * grad_b[i]
            
        # Log training metrics
        loss = np.mean((y.reshape(-1, 1) - output)**2)
        return loss, output

    def predict(self, X):
        """Predict class labels (0 or 1) for the given input data."""
        output = self.forward(X)
        return np.where(output >= 0.5, 1, 0).flatten()
