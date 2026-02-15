# Neural Network Learning Visualizer

This application simulates and visualizes the training process of Perceptron, Adaline, and Multi-Layer Perceptron (MLP) networks.

## Setup

1.  Ensure you have Python installed.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Run the main script:
```bash
python main.py
```

## Features

-   **Algorithms**: Perceptron, Adaline, MLP (Backpropagation).
-   **Visualizations**:
    -   Real-time Decision Boundary updates.
    -   Network Architecture Diagram (updates weights visualization).
    -   Loss/Accuracy Metrics plot.
-   **Datasets**: Linear, Circles, Moons, Spirals, Logic Gates (AND/OR/XOR), and **Manual Entry** (Click to add points).
-   **Controls**: Adjustable learning rate, epochs, hidden layers, and activation functions.

## Usage Tips

-   **Manual Data**: Select "Manual" from the Dataset dropdown. Left-click on the plot to add Class 0 (Red) points, Right-click to add Class 1 (Blue) points.
-   **MLP Config**: Enter hidden layers as comma-separated values, e.g., `4,4` for two hidden layers with 4 neurons each.
-   **Reset**: Use the Reset button to clear the current simulation and start fresh.
-   **XOR Problem**: Try training a standard Perceptron on XOR data to see it fail, then switch to MLP with at least one hidden layer (e.g., `3`) to see it succeed!
