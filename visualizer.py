import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Visualizer:
    """
    A utility class for rendering machine learning results using Matplotlib.
    Handles decision boundary plotting and training metric visualizations.
    """
    def __init__(self, ax):
        """
        Initialize the Visualizer.

        Args:
            ax (matplotlib.axes.Axes): The axes object where plots will be drawn.
        """
        self.ax = ax
    
    def plot_decision_regions(self, X, y, classifier, resolution=0.02, test_idx=None):
        """
        Plot the decision regions and training samples on a 2D plane.

        Args:
            X (ndarray): Feature matrix.
            y (ndarray): Target vector.
            classifier (object): An object with a `predict` method.
            resolution (float): The step size for the meshgrid (smaller = smoother).
            test_idx (optional): Indices of samples to highlight as test data.
        """
        self.ax.clear()
        
        # Setup marker generator and vibrant color map for dark theme
        markers = ('o', 's', '^', 'v', '<')
        colors = ('#ff3333', '#33adff', '#33ff33', '#aaaaaa', '#00ffff') # Red, Blue, Green, Gray, Cyan
        
        unique_labels = np.unique(y)
        cmap = ListedColormap(colors[:len(unique_labels)])

        # 1. Define the plot boundaries with a small margin
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        # 2. Create a meshgrid to cover the entire feature space
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        
        # 3. Predict the class for every point in the meshgrid to find the boundary
        try:
            # Flat array of meshgrid points
            grid_points = np.array([xx1.ravel(), xx2.ravel()]).T
            Z = classifier.predict(grid_points)
            Z = Z.reshape(xx1.shape)
            
            # Draw the filled contour (the background color bands)
            self.ax.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        except Exception as e:
            # During the very first step, the classifier might not be initialized
            pass

        # 4. Set axes limits
        self.ax.set_xlim(xx1.min(), xx1.max())
        self.ax.set_ylim(xx2.min(), xx2.max())

        # 5. Plot actual data samples as scatter points
        for idx, cl in enumerate(unique_labels):
            self.ax.scatter(x=X[y == cl, 0], 
                            y=X[y == cl, 1],
                            alpha=0.9, 
                            c=colors[idx],
                            marker=markers[idx], 
                            label=f'Class {cl}', 
                            edgecolor='white', 
                            s=60) # Slightly larger for better visibility
        
        # Labeling and styling for dark mode
        self.ax.set_xlabel('Feature 1', color='white')
        self.ax.set_ylabel('Feature 2', color='white')
        self.ax.tick_params(colors='white')
        
        # Professional legend styling
        legend = self.ax.legend(loc='upper left', frameon=True)
        frame = legend.get_frame()
        frame.set_facecolor('#2b2b2b')
        frame.set_edgecolor('#555555')
        for text in legend.get_texts():
            text.set_color("white")
            
        # Subtle grid for reference
        self.ax.grid(True, color='#555555', alpha=0.5, linestyle='--')

    def plot_metrics(self, history_ax, history, metric='loss'):
        """
        Plot training metrics (Loss or Accuracy) over time.

        Args:
            history_ax (matplotlib.axes.Axes): Axes for the metric plot.
            history (dict): Dictionary containing the history of the metric.
            metric (str): Name of the metric to plot ('loss' or 'accuracy').
        """
        history_ax.clear()
        
        if metric in history and len(history[metric]) > 0:
             # Cyan for accuracy, Pink/Red for loss
             plot_color = '#00ffcc' if metric == 'accuracy' else '#ff3366'
             
             epochs = range(1, len(history[metric]) + 1)
             history_ax.plot(epochs, history[metric], 
                             marker='o', color=plot_color, 
                             markersize=4, linewidth=2)
             
             history_ax.set_xlabel('Epochs', color='white')
             history_ax.set_ylabel(metric.capitalize(), color='white')
             history_ax.tick_params(colors='white')
             history_ax.grid(True, color='#555555', alpha=0.5, linestyle='--')
