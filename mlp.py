import numpy as np
from utils import sigmoid, sigmoid_prime, tanh, tanh_prime, relu, relu_prime, softmax

class MLP:
    """Multi-Layer Perceptron with Backpropagation."""
    def __init__(self, layers, activation='sigmoid', eta=0.01, epochs=100, random_state=1):
        """
        layers: list of integers, e.g. [2, 4, 1] (input, hidden, output)
        """
        self.layers = layers
        self.activation_name = activation
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.init_weights()
        
        self.history = {'loss': [], 'accuracy': []}
        
    def init_weights(self):
        rng = np.random.RandomState(self.random_state)
        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            # He initialization for ReLU, Xavier/Glorot for Sigmoid/Tanh
            if self.activation_name == 'relu':
                std = np.sqrt(2.0 / self.layers[i])
            else:
                std = np.sqrt(1.0 / self.layers[i])
                
            w = rng.normal(0.0, std, (self.layers[i], self.layers[i+1]))
            b = np.zeros((1, self.layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def _activation(self, z):
        if self.activation_name == 'sigmoid':
            return sigmoid(z)
        elif self.activation_name == 'tanh':
            return tanh(z)
        elif self.activation_name == 'relu':
            return relu(z)
        return z

    def _activation_prime(self, z):
        if self.activation_name == 'sigmoid':
            return sigmoid_prime(z)
        elif self.activation_name == 'tanh':
            return tanh_prime(z)
        elif self.activation_name == 'relu':
            return relu_prime(z)
        return np.ones_like(z)

    def forward(self, X):
        """Forward pass. Returns activations and z-values for all layers."""
        self.a = [X] # Activations
        self.z = []  # Net inputs

        for i in range(len(self.weights)):
            net_input = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.z.append(net_input)
            
            # Last layer usually sigmoid for binary classification
            if i == len(self.weights) - 1:
                # Assuming binary classification for now
                activation = sigmoid(net_input) 
            else:
                activation = self._activation(net_input)
            self.a.append(activation)
        
        return self.a[-1]

    def backward(self, X, y):
        """Backpropagation to compute gradients."""
        m = X.shape[0]
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        # Output layer error
        # Assuming Mean Squared Error for simplicity in derivation visualization
        # loss = (a - y)^2 -> d_loss/da = 2(a - y)
        # d_loss/dz = d_loss/da * da/dz
        # For sigmoid output: da/dz = a(1-a)
        # So delta = (a - y) if using Cross Entropy with Logits... but let's stick to simple MSE/Sigmoid for now
        
        # dC/dz_L = (a_L - y) * sigma'(z_L)
        # If we use MSE: C = 0.5(y-a)^2, dC/da = (a-y)
        # delta_L = (a_L - y) * sigmoid_prime(z_L)
        
        delta = (self.a[-1] - y.reshape(-1, 1)) * sigmoid_prime(self.z[-1])
        
        grad_w[-1] = np.dot(self.a[-2].T, delta)
        grad_b[-1] = np.sum(delta, axis=0, keepdims=True)

        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[i+1].T) * self._activation_prime(self.z[i])
            grad_w[i] = np.dot(self.a[i].T, delta)
            grad_b[i] = np.sum(delta, axis=0, keepdims=True)
            
        return grad_w, grad_b

    def train_step(self, X, y):
        """Perform one training step (one epoch or batch)."""
        # Forward
        output = self.forward(X)
        
        # Backward
        grad_w, grad_b = self.backward(X, y)
        
        # Update
        for i in range(len(self.weights)):
            self.weights[i] -= self.eta * grad_w[i]
            self.biases[i] -= self.eta * grad_b[i]
            
        loss = np.mean((y.reshape(-1, 1) - output)**2)
        return loss, output

    def predict(self, X):
        output = self.forward(X)
        return np.where(output >= 0.5, 1, 0).flatten()
