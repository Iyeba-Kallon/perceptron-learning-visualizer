import numpy as np

class AdalineGD:
    """Adaptive Linear Neuron Classifier."""
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = None
        self.b_ = None
        self.cost_ = []
        self.history = []

    def fit(self, X, y):
        """ Fit training data. """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.cost_ = []
        self.history = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            
            # Gradient descent update
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            
            cost = (errors**2).mean()
            self.cost_.append(cost)
            
            # Save history for visualization
            self.history.append({
                'w': self.w_.copy(),
                'b': self.b_,
                'cost': cost,
                'epoch': i
            })
        return self

    def net_input(self, X):
        """Calculate net input."""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation."""
        return X  # Linear activation

    def predict(self, X):
        """Return class label after unit step."""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0) # Threshold at 0.5 for binary
    
    def train_step(self, X, y):
        """Perform a single batch gradient descent step."""
        if self.w_ is None:
             rgen = np.random.RandomState(self.random_state)
             self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
             self.b_ = np.float_(0.)
             
        net_input = self.net_input(X)
        output = self.activation(net_input)
        errors = (y - output)
        
        # Batch update
        self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
        self.b_ += self.eta * 2.0 * errors.mean()
        
        cost = (errors**2).mean()
        return cost, output
