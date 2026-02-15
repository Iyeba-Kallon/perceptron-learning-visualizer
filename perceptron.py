import numpy as np

class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = None
        self.b_ = None
        self.errors_ = []
        self.history = []

    def fit(self, X, y):
        """Fit training data."""
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []
        self.history = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # Save state before update for visualization
                self.history.append({
                    'w': self.w_.copy(),
                    'b': self.b_,
                    'x': xi,
                    'target': target
                })
                
                update = self.eta * (target - self.predict_single(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input."""
        return np.dot(X, self.w_) + self.b_

    def predict_single(self, x):
        """Predict class label for single sample."""
        return np.where(self.net_input(x) >= 0.0, 1, 0)
    
    def predict(self, X):
        """Return class label after unit step."""
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def train_step(self, x, y):
        """Perform a single training step (online learning)."""
        if self.w_ is None:
             rgen = np.random.RandomState(self.random_state)
             self.w_ = rgen.normal(loc=0.0, scale=0.01, size=x.shape[0])
             self.b_ = np.float_(0.)
        
        y_pred = self.predict_single(x)
        error = y - y_pred
        update = self.eta * error
        
        self.w_ += update * x
        self.b_ += update
        
        loss = int(error != 0.0)
        return loss, y_pred
