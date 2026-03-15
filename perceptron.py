import numpy as np

class Perceptron:
    """
    A simple Perceptron classifier based on the classic 1957 model.
    It performs binary classification by learning a linear weight vector.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """
        Initialize the Perceptron.

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
        self.errors_ = []  # Misclassifications in each epoch
        self.history = []  # Detailed step-by-step history for visualization

    def fit(self, X, y):
        """
        Fit training data to the model.

        Args:
            X (array-like): Training vectors, shape [n_samples, n_features].
            y (array-like): Target values, shape [n_samples].

        Returns:
            self : object
        """
        rgen = np.random.RandomState(self.random_state)
        # Initialize weights from a normal distribution for symmetry breaking
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.errors_ = []
        self.history = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # We store the state *before* the update for smooth animation in the UI
                self.history.append({
                    'w': self.w_.copy(),
                    'b': self.b_,
                    'x': xi,
                    'target': target
                })
                
                # Perceptron update rule: w = w + eta * (y - y_pred) * x
                update = self.eta * (target - self.predict_single(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate the net input (dot product of input and weights plus bias)."""
        return np.dot(X, self.w_) + self.b_

    def predict_single(self, x):
        """Predict class label for a single sample (unit step function)."""
        return np.where(self.net_input(x) >= 0.0, 1, 0)
    
    def predict(self, X):
        """Return class labels for an array of samples after applying the step function."""
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def train_step(self, x, y):
        """
        Perform a single training step (Online Learning mode).
        This is used by the GUI to animate the learning process one sample at a time.
        """
        if self.w_ is None:
             rgen = np.random.RandomState(self.random_state)
             self.w_ = rgen.normal(loc=0.0, scale=0.01, size=x.shape[0])
             self.b_ = np.float64(0.0)
        
        y_pred = self.predict_single(x)
        error = y - y_pred
        update = self.eta * error
        
        # In-place update of weights and bias
        self.w_ += update * x
        self.b_ += update
        
        # Loss is 1 if there was a misclassification, 0 otherwise
        loss = int(error != 0.0)
        return loss, y_pred
