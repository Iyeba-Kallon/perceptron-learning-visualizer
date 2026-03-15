import matplotlib.pyplot as plt
import numpy as np

class NetworkDrawer:
    """
    Renders a schematic diagram of a Neural Network's architecture.
    Visualizes layer connectivity, neuron positions, and weight intensities.
    """
    def __init__(self, ax):
        """
        Initialize the NetworkDrawer.

        Args:
            ax (matplotlib.axes.Axes): The axes object where the network will be drawn.
        """
        self.ax = ax
        self.node_positions = []
        
    def draw(self, layers, weights, biases=None, activations=None):
        """
        Draw the neural network architecture based on layer config and weights.

        Args:
            layers (list): Number of neurons in each layer, e.g., [2, 4, 1].
            weights (list): List of numpy arrays representing weight matrices.
            biases (list, optional): Bias vectors for each layer.
            activations (list, optional): Current activation values for display.
        """
        self.ax.clear()
        self.ax.axis('off')  # Hide axes for a clean diagram
        
        # 1. Coordinate setup for the diagram (relative 0.0 to 1.0)
        left, right = 0.1, 0.9
        bottom, top = 0.1, 0.9
        
        # Calculate horizontal and vertical spacing
        v_spacing = (top - bottom) / float(max(layers))
        h_spacing = (right - left) / float(len(layers) - 1)
        
        # 2. Iterate through layers to calculate node positions and draw neurons
        self.node_positions = []
        for i, n_neurons in enumerate(layers):
            layer_nodes = []
            # Vertically center each layer
            layer_top = top - (max(layers) - n_neurons) * v_spacing / 2.0
            
            for j in range(n_neurons):
                x = left + i * h_spacing
                y = layer_top - j * v_spacing
                layer_nodes.append((x, y))
                
                # Draw Neuron (Circle)
                # Outer circle for z-order layering
                neuron = plt.Circle((x, y), 0.04, color='white', ec='black', zorder=4)
                self.ax.add_artist(neuron)
                
            self.node_positions.append(layer_nodes)
            
        # 3. Draw Synapses (Edges) connecting neurons
        # We use color and thickness to represent the "weight" of each connection
        for i in range(len(layers) - 1):
            if i >= len(weights):
                break
            
            w_matrix = weights[i]
            # Find the max weight for normalization (avoid division by zero)
            w_max = np.max(np.abs(w_matrix)) if np.max(np.abs(w_matrix)) > 0 else 1.0
            
            for j in range(layers[i]):      # Source neuron index
                for k in range(layers[i+1]):  # Target neuron index
                    x1, y1 = self.node_positions[i][j]
                    x2, y2 = self.node_positions[i+1][k]
                    
                    weight_val = w_matrix[j, k]
                    
                    # Coding: Cyan = Positive Correlation, Red = Negative Correlation
                    line_color = '#00ffcc' if weight_val >= 0 else '#ff3366'
                    
                    # Normalize opacity and width based on weight strength
                    # We add a base alpha so thin lines don't completely disappear
                    alpha = max(min(abs(weight_val) / w_max, 1.0), 0.2)
                    line_width = abs(weight_val) / w_max * 3.0 + 0.5 
                    
                    # Draw actual connection line
                    self.ax.plot([x1, x2], [y1, y2], 
                                 c=line_color, 
                                 linewidth=line_width, 
                                 alpha=alpha, 
                                 zorder=1)
