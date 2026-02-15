import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import threading
import time

# Import our modules
from perceptron import Perceptron
from adaline import AdalineGD
from mlp import MLP
from visualizer import Visualizer
from network_drawer import NetworkDrawer
import data_generator
from utils import logic_gate_data, accuracy, mse

class NeuralNetSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Neural Network Learning Simulator")
        self.root.geometry("1600x900")
        
        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # --- Main Layout ---
        self.create_menu()
        
        self.main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.left_panel = ttk.Frame(self.main_pane, width=300)
        self.main_pane.add(self.left_panel, weight=1)
        
        self.center_panel = ttk.Frame(self.main_pane, width=800)
        self.main_pane.add(self.center_panel, weight=4)
        
        self.right_panel = ttk.Frame(self.main_pane, width=300)
        self.main_pane.add(self.right_panel, weight=1)
        
        self.setup_left_panel()
        self.setup_center_panel()
        self.setup_right_panel()
        
        # Controller State
        self.is_playing = False
        self.current_epoch = 0
        self.algorithm = None
        self.X = None
        self.y = None
        self.history = {'loss': [], 'accuracy': []}
        
        # Visualizers (Init with placeholder axes, will be set in setup_center_panel)
        # We need to initialize them AFTER setup_center_panel creates the axes
        self.vis = Visualizer(self.ax_boundary)
        self.net_drawer = NetworkDrawer(self.ax_network)
        
        # Connect click event
        self.canvas_boundary.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes != self.ax_boundary:
            return
        if self.dataset_var.get() != "Manual":
            return
            
        # Left click: Class 0, Right click: Class 1
        # Matplotlib event.button: 1=Left, 3=Right
        label = 0 if event.button == 1 else 1
        
        # Coordinates
        x, y = event.xdata, event.ydata
        if x is None or y is None: return
        
        new_point = np.array([[x, y]])
        new_label = np.array([label])
        
        if self.X is None:
            self.X = new_point
            self.y = new_label
        else:
            self.X = np.vstack([self.X, new_point])
            self.y = np.hstack([self.y, new_label])
            
        self.log_message(f"Added point ({x:.2f}, {y:.2f}) -> Class {label}")
        
        # Re-plot
        self.ax_boundary.clear()
        markers = ('o', 's')
        colors = ('red', 'blue')
        
        # Helper to plot raw data
        for idx, cl in enumerate(np.unique(self.y)):
            # Ensure safe indexing into colors/markers
            c_idx = int(cl) % len(colors)
            m_idx = int(cl) % len(markers)
            
            self.ax_boundary.scatter(x=self.X[self.y == cl, 0], 
                            y=self.X[self.y == cl, 1],
                            alpha=0.8, 
                            c=colors[c_idx],
                            marker=markers[m_idx], 
                            label=f'Class {cl}', 
                            edgecolor='black')
        
        self.ax_boundary.set_xlim(min(self.X[:,0])-1, max(self.X[:,0])+1)
        self.ax_boundary.set_ylim(min(self.X[:,1])-1, max(self.X[:,1])+1)
        self.ax_boundary.legend(loc='upper left')
        self.canvas_boundary.draw()
        
    def create_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)

    def setup_left_panel(self):
        # -- Algorithm Selection --
        algo_labelframe = ttk.LabelFrame(self.left_panel, text="Algorithm Selection")
        algo_labelframe.pack(fill=tk.X, padx=5, pady=5)
        
        self.algo_var = tk.StringVar(value="Perceptron")
        ttk.Radiobutton(algo_labelframe, text="Perceptron", variable=self.algo_var, value="Perceptron").pack(anchor=tk.W)
        ttk.Radiobutton(algo_labelframe, text="Adaline", variable=self.algo_var, value="Adaline").pack(anchor=tk.W)
        ttk.Radiobutton(algo_labelframe, text="Multi-Layer Perceptron (MLP)", variable=self.algo_var, value="MLP").pack(anchor=tk.W)
        
        # -- Hyperparameters --
        hp_labelframe = ttk.LabelFrame(self.left_panel, text="Hyperparameters")
        hp_labelframe.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(hp_labelframe, text="Learning Rate (η):").pack(anchor=tk.W)
        self.lr_scale = tk.Scale(hp_labelframe, from_=0.0001, to=1.0, resolution=0.001, orient=tk.HORIZONTAL)
        self.lr_scale.set(0.01)
        self.lr_scale.pack(fill=tk.X)
        
        ttk.Label(hp_labelframe, text="Epochs:").pack(anchor=tk.W)
        self.epochs_entry = ttk.Entry(hp_labelframe)
        self.epochs_entry.insert(0, "50")
        self.epochs_entry.pack(fill=tk.X)
        
        # -- MLP Specific --
        mlp_labelframe = ttk.LabelFrame(self.left_panel, text="MLP Config")
        mlp_labelframe.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(mlp_labelframe, text="Hidden Layers (e.g., 4):").pack(anchor=tk.W)
        self.hidden_layers_entry = ttk.Entry(mlp_labelframe)
        self.hidden_layers_entry.insert(0, "4")
        self.hidden_layers_entry.pack(fill=tk.X)
        
        ttk.Label(mlp_labelframe, text="Activation:").pack(anchor=tk.W)
        self.activation_var = tk.StringVar(value="Sigmoid")
        activations = ["Sigmoid", "Tanh", "ReLU"]
        ttk.OptionMenu(mlp_labelframe, self.activation_var, activations[0], *activations).pack(fill=tk.X)

        # -- Dataset --
        data_labelframe = ttk.LabelFrame(self.left_panel, text="Dataset")
        data_labelframe.pack(fill=tk.X, padx=5, pady=5)
        
        self.dataset_var = tk.StringVar(value="Linear")
        datasets = ["Linear", "AND", "OR", "XOR", "Circles", "Moons", "Spiral", "Manual"]
        ttk.OptionMenu(data_labelframe, self.dataset_var, datasets[0], *datasets).pack(fill=tk.X)
        
        ttk.Button(data_labelframe, text="Generate/Load Data", command=self.generate_data).pack(fill=tk.X, pady=5)
        ttk.Label(data_labelframe, text="(Manual: L-Click=0, R-Click=1)", font=("Arial", 8)).pack(anchor=tk.W)


    def setup_center_panel(self):
        # Tabs for Decision Boundary, Network, Loss
        self.notebook = ttk.Notebook(self.center_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Decision Boundary
        self.tab_boundary = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_boundary, text="Decision Boundary")
        
        self.fig_boundary = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_boundary = self.fig_boundary.add_subplot(111)
        self.canvas_boundary = FigureCanvasTkAgg(self.fig_boundary, self.tab_boundary)
        self.canvas_boundary.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 2: Network Architecture
        self.tab_network = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_network, text="Network Architecture")
        
        self.fig_network = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_network = self.fig_network.add_subplot(111)
        self.canvas_network = FigureCanvasTkAgg(self.fig_network, self.tab_network)
        self.canvas_network.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab 3: History
        self.tab_loss = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_loss, text="Metrics")
        
        self.fig_loss = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_loss = self.fig_loss.add_subplot(111)
        self.canvas_loss = FigureCanvasTkAgg(self.fig_loss, self.tab_loss)
        self.canvas_loss.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Playback Controls
        controls_frame = ttk.Frame(self.center_panel)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_start = ttk.Button(controls_frame, text="Start Training", command=self.start_training)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = ttk.Button(controls_frame, text="Stop", command=self.stop_training, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(controls_frame, text="Reset", command=self.reset_simulation).pack(side=tk.LEFT, padx=5)

    def setup_right_panel(self):
        stats_labelframe = ttk.LabelFrame(self.right_panel, text="Training Stats")
        stats_labelframe.pack(fill=tk.X, padx=5, pady=5)
        
        self.epoch_label = ttk.Label(stats_labelframe, text="Epoch: 0")
        self.epoch_label.pack(anchor=tk.W)
        
        self.loss_label = ttk.Label(stats_labelframe, text="Loss: N/A")
        self.loss_label.pack(anchor=tk.W)
        
        self.acc_label = ttk.Label(stats_labelframe, text="Accuracy: N/A")
        self.acc_label.pack(anchor=tk.W)
        
        log_labelframe = ttk.LabelFrame(self.right_panel, text="Event Log")
        log_labelframe.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_labelframe, height=20, width=30, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log_message(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def show_about(self):
        messagebox.showinfo("About", "Neural Network Learning Simulator v1.0")
        
    def generate_data(self):
        selection = self.dataset_var.get()
        self.log_message(f"Generating {selection} dataset...")
        
        if selection == "Manual":
             self.log_message("Manual mode: Click on plot to add points.")
             self.X = None
             self.y = None
             self.ax_boundary.clear()
             self.canvas_boundary.draw()
             return

        if selection == "Linear":
            self.X, self.y = data_generator.generate_linear_data()
        elif selection == "Circles":
            self.X, self.y = data_generator.generate_circles_data()
        elif selection == "Moons":
            self.X, self.y = data_generator.generate_moons_data()
        elif selection == "Spiral":
            self.X, self.y = data_generator.generate_spiral_data()
        elif selection == "AND":
             self.X, self.y = logic_gate_data("AND")
        elif selection == "OR":
             self.X, self.y = logic_gate_data("OR")
        elif selection == "XOR":
             self.X, self.y = logic_gate_data("XOR")
        
        # Plot initial data
        self.ax_boundary.clear()
        
        # Helper to plot raw data before classifier is ready
        markers = ('o', 's')
        colors = ('red', 'blue')
        # Handle binary classification for now
        unique_y = np.unique(self.y)
        for idx, cl in enumerate(unique_y):
            self.ax_boundary.scatter(x=self.X[self.y == cl, 0], 
                            y=self.X[self.y == cl, 1],
                            alpha=0.8, 
                            c=colors[idx % len(colors)],
                            marker=markers[idx % len(markers)], 
                            label=f'Class {cl}', 
                            edgecolor='black')
        self.ax_boundary.legend(loc='upper left')
        self.canvas_boundary.draw()
        
        self.log_message(f"Data generated. Samples: {len(self.y)}")

    def init_algorithm(self):
        algo_name = self.algo_var.get()
        lr = self.lr_scale.get()
        try:
             epochs = int(self.epochs_entry.get())
        except ValueError:
             messagebox.showerror("Error", "Invalid Epochs.")
             return False
        
        if algo_name == "Perceptron":
            self.algorithm = Perceptron(eta=lr, n_iter=epochs)
        elif algo_name == "Adaline":
            self.algorithm = AdalineGD(eta=lr, n_iter=epochs)
        elif algo_name == "MLP":
            hidden_str = self.hidden_layers_entry.get()
            try:
                # Parse hidden layers "4,4" -> [2, 4, 4, 1]
                hidden = [int(x) for x in hidden_str.split(',') if x.strip()]
            except ValueError:
                messagebox.showerror("Error", "Invalid Hidden Layers format. Use comma separated numbers like '4,4'")
                return False
                
            input_dim = self.X.shape[1]
            output_dim = 1 # Binary classification
            layers = [input_dim] + hidden + [output_dim]
            
            activation = self.activation_var.get().lower()
            self.algorithm = MLP(layers=layers, activation=activation, eta=lr, epochs=epochs)
            
            # Draw initial network
            self.net_drawer.draw(layers, self.algorithm.weights)
            self.canvas_network.draw()
            
        return True

    def start_training(self):
        if self.X is None:
            messagebox.showwarning("Warning", "Please generate data first!")
            return
            
        if self.is_playing:
            return

        if not self.init_algorithm():
            return
            
        self.is_playing = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.current_epoch = 0
        self.history = {'loss': [], 'accuracy': []}
        
        self.log_message(f"Starting training with {self.algo_var.get()}...")
        self.train_loop()
        
    def stop_training(self):
        self.is_playing = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.log_message("Training stopped.")

    def reset_simulation(self):
        self.stop_training()
        self.algorithm = None
        self.current_epoch = 0
        self.history = {'loss': [], 'accuracy': []}
        self.ax_boundary.clear()
        self.ax_network.clear()
        self.ax_loss.clear()
        self.canvas_boundary.draw()
        self.canvas_network.draw()
        self.canvas_loss.draw()
        self.log_message("Simulation Reset.")
        self.epoch_label.config(text="Epoch: 0")
        self.loss_label.config(text="Loss: N/A")
        self.acc_label.config(text="Accuracy: N/A")
        
        if self.X is not None and self.dataset_var.get() != "Manual":
             self.generate_data() # Re-plot data
        elif self.dataset_var.get() == "Manual":
             self.X = None
             self.y = None # Reset manual data

    def train_loop(self):
        if not self.is_playing:
            return
            
        max_epochs = int(self.epochs_entry.get())
        if self.current_epoch >= max_epochs:
            self.stop_training()
            self.log_message("Max epochs reached.")
            return
            
        # Perform one "step" (epoch)
        loss = 0
        acc = 0
        
        # Different handling for MLP vs others
        if isinstance(self.algorithm, MLP):
            loss, _ = self.algorithm.train_step(self.X, self.y)
            pred = self.algorithm.predict(self.X)
            acc = accuracy(self.y.reshape(-1, 1), pred.reshape(-1, 1))
            
            # Update network drawer with new weights
            self.ax_network.clear()
            self.net_drawer.draw(self.algorithm.layers, self.algorithm.weights)
            self.canvas_network.draw()
            
        elif isinstance(self.algorithm, AdalineGD):
            loss, _ = self.algorithm.train_step(self.X, self.y)
            pred = self.algorithm.predict(self.X)
            acc = accuracy(self.y, pred)
            
        elif isinstance(self.algorithm, Perceptron):
            # Perceptron train_step (Stochastic loop)
            # We will run one full pass over data for "one epoch" visualization
            errors = 0
            # Shuffle for stochastic
            r = np.random.permutation(len(self.y))
            for i in r:
                l, _ = self.algorithm.train_step(self.X[i], self.y[i])
                errors += l
            loss = errors # Perceptron uses misclassifications as "loss" proxy here
            pred = self.algorithm.predict(self.X)
            acc = accuracy(self.y, pred)

        self.current_epoch += 1
        self.history['loss'].append(loss)
        self.history['accuracy'].append(acc)
        
        # Update UI
        self.epoch_label.config(text=f"Epoch: {self.current_epoch}/{max_epochs}")
        self.loss_label.config(text=f"Loss: {loss:.4f}")
        self.acc_label.config(text=f"Accuracy: {acc*100:.1f}%")
        
        # Update Plots (every 5 epochs to speed up animation if needed, or every 1)
        # Using every epoch for smooth vis
        self.vis.plot_decision_regions(self.X, self.y, self.algorithm)
        self.canvas_boundary.draw()
        
        self.vis.plot_metrics(self.ax_loss, self.history, metric='loss')
        self.canvas_loss.draw()
        
        # Schedule next loop
        self.root.after(50, self.train_loop) # 50ms delay
