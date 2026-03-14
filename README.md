# Neural Network Learning Visualizer

This application is an interactive, visually stunning desktop tool that simulates and visualizes the training process of foundational neural network algorithms. It is designed to demystify how models like the Perceptron, Adaline, and Multi-Layer Perceptrons (MLP) shift their decision boundaries step-by-step during the learning process.

Recently overhauled with a modern dark-themed CustomTkinter UI and vibrant animations, this sandbox environment turns abstract calculus into an interactive experience.

## Features

- **Algorithms Built from Scratch**:
  - **Perceptron**: The classic linear classifier. Watch it solve linearly separable problems and fail on non-linear ones like XOR.
  - **Adaline (Adaptive Linear Neuron)**: Uses Mean Squared Error and Gradient Descent for a smoother, mathematically grounded approach to finding optimal weights.
  - **Multi-Layer Perceptron (MLP)**: Solves non-linear problems using customizable hidden layers, backpropagation, and non-linear activation functions (`Sigmoid`, `Tanh`, `ReLU`).
- **Interactive Visualizations (Real-Time)**:
  - **Decision Boundary**: A dynamic contour map showing exactly how the model divides the 2D space epoch by epoch.
  - **Network Architecture Graphic**: Dynamically draws the network graph. Synapses light up with vibrant glowing colors (cyan for positive weights, red for negative weights) and scale in thickness based on weight magnitude!
  - **Metrics Plot**: A live tracker for Loss and Accuracy to monitor convergence and overfitting.
- **Dynamic Datasets**:
  - Built-in generators for Linear, Circles, Moons, Spirals, and classic Logic Gates (AND/OR/XOR).
  - **Manual Entry Mode**: Literally click on the canvas to place custom data points and challenge the model to separate them! (Left-click for Class 0, Right-click for Class 1).

## Installation & Setup

1. **Prerequisites**: Ensure you have Python 3.8+ installed on your system.
2. **Clone the repository**:
   ```bash
   git clone https://github.com/Iyeba-Kallon/neural-network-learning-visualizer.git
   cd neural-network-learning-visualizer
   ```
3. **Install Dependencies**:
   The application requires several libraries, including `customtkinter` for the modern UI, and `numpy`/`matplotlib` for the math and plotting. Install them via:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Launch the visualizer by running the main Python script:
```bash
python main.py
```

## How to Use the Visualizer

1. **Select an Algorithm**: Choose between Perceptron, Adaline, or MLP from the left panel.
2. **Configure Hyperparameters**: Adjust the Learning Rate (η) and the maximum number of Epochs. If using an MLP, specify the hidden layers (e.g., `4,4` for two layers with 4 neurons each) and choose an activation function.
3. **Generate Data**: Select a dataset type from the dropdown. 
   - *Tip*: Select **Manual** mode to plot your own data. Left-click anywhere on the "Decision Boundary" plot to place a Class 0 (Red) point, and Right-click to place a Class 1 (Blue) point.
4. **Train**: Click **Start Training**. Watch the three right-hand panes animate in real-time as the model attempts to classify your data!
5. **Reset & Experiment**: Use the **Stop** and **Reset** buttons to halt training or start over with different configurations. Try training a standard Perceptron on XOR data to see it fail, then switch to MLP with at least one hidden layer to see it succeed!

## Future Enhancements / To-Do

- [ ] Connect the output of the Network Drawer activation visualizer to show real-time neuron firing intensity.
- [ ] Add more complex optimizers for MLP (e.g., Adam, RMSprop).
- [ ] Implement multi-class classification (currently binary classification).
- [ ] Export trained weights/models capability.
