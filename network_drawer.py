import matplotlib.pyplot as plt
import numpy as np

class NetworkDrawer:
    def __init__(self, ax):
        self.ax = ax
        
    def draw(self, layers, weights, biases=None, activations=None):
        """
        Draw neural network architecture.
        layers: list of int, e.g. [2, 3, 1]
        weights: list of arrays
        """
        self.ax.clear()
        self.ax.axis('off')
        
        left = 0.1
        right = 0.9
        bottom = 0.1
        top = 0.9
        
        v_spacing = (top - bottom) / float(max(layers))
        h_spacing = (right - left) / float(len(layers) - 1)
        
        # Calculate node positions
        self.node_positions = []
        for i, n in enumerate(layers):
            layer_nodes = []
            layer_top = top - (max(layers) - n) * v_spacing / 2.0
            for j in range(n):
                x = left + i * h_spacing
                y = layer_top - j * v_spacing
                layer_nodes.append((x, y))
                
                # Draw Node
                circle = plt.Circle((x, y), 0.04, color='white', ec='black', zorder=4)
                self.ax.add_artist(circle)
                
                # Draw Activation value if provided
                if activations and i < len(activations):
                    # For input layer (i=0), activations[0] might be the input X
                    # This needs to be handled carefully in the main loop
                    pass
                    
            self.node_positions.append(layer_nodes)
            
        # Draw edges (Weights)
        # weights[i] connects layer i to i+1
        # dimension of weights[i] is (nodes_in, nodes_out) usually
        for i in range(len(layers) - 1):
            if i >= len(weights): break
            
            w = weights[i]
            # w shape: (n_current, n_next)
            
            # Normalize weights for thickness/color
            w_max = np.max(np.abs(w)) if np.max(np.abs(w)) > 0 else 1.0
            
            for j in range(layers[i]): # Source node
                for k in range(layers[i+1]): # Target node
                    x1, y1 = self.node_positions[i][j]
                    x2, y2 = self.node_positions[i+1][k]
                    
                    weight_val = w[j, k]
                    color = 'red' if weight_val > 0 else 'blue'
                    alpha = min(abs(weight_val) / w_max, 1.0)
                    width = abs(weight_val) / w_max * 2.0 + 0.1
                    
                    self.ax.plot([x1, x2], [y1, y2], c=color, linewidth=width, alpha=alpha, zorder=1)
