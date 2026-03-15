import numpy as np

class AdalineGD:
    """
    Adaptive Linear Neuron (ADAptive LInear NEuron - ADAline) Classifier.
    
    Unlike the Perceptron, Adaline uses a linear activation function during 
    the learning phase, allowing the use of Gradient Descent to minimize 
    a continuous cost function (Mean Squared Error).
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """
        Initialize the Adaline model.

        Args:
            eta (float): Learning rate (between 0.0 and 1.0).
            n_iter (int): Number of passes over the training dataset.
            random_state (int): Seed for random weight initialization.
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = None  # Weights after fitting
        self.b_ = None  # Bias unit after fitting
        self.cost_ = []  # Average squared error (cost) in each epoch
        self.history = []  # Training history for visualization purposes

    def fit(self, X, y):
        """
        Fit training data using Batch Gradient Descent.

        Args:
            X (array-like): Training vectors, shape [n_samples, n_features].
            y (array-like): Target values, shape [n_samples].

        Returns:
            self : object
        """
        rgen = np.random.RandomState(self.random_state)
        # Initialize weights with small random values to break symmetry
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.cost_ = []
        self.history = []

        for i in range(self.n_iter):
            # Calculate the linear activation output: net_input = X*w + b
            net_input = self.net_input(X)
            output = self.activation(net_input)
            
            # Compute the error (difference between target and continuous output)
            errors = (y - output)
            
            # Update weights and bias using the gradient of the MSE cost function
            # Formula: w = w + eta * 2/n * sum((y - output) * x)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            
            # Calculate the Sum Squared Error (SSE) or Mean Squared Error (MSE)
            cost = (errors**2).mean()
            self.cost_.append(cost)
            
            # Capture state for visualization
            self.history.append({
                'w': self.w_.copy(),
                'b': self.b_,
                'cost': cost,
                'epoch': i
            })
        return self

    def net_input(self, X):
        """Calculate the net input (weighted sum)."""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """
        Compute the linear activation. 
        In Adaline, the activation function is simply the identity function.
        """
        return X

    def predict(self, X):
        """
        Return class labels after unit step function.
        We apply a threshold (0.5 for binary) to the linear activation.
        """
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
    def train_step(self, X, y):
        """
        Perform a single batch gradient descent step.
        Used by the GUI to step through the training process.
        """
        if self.w_ is None:
             rgen = np.random.RandomState(self.random_state)
             self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
             self.b_ = np.float64(0.0)
             
        net_input = self.net_input(X)
        output = self.activation(net_input)
        errors = (y - output)
        
        # Perform one Batch update iteration
        self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
        self.b_ += self.eta * 2.0 * errors.mean()
        
        # Calculate current cost (MSE)
        cost = (errors**2).mean()
        return cost, output
