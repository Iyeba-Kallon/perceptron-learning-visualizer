import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import time

# Import our modules
# (We will import them inside the controller or methods to avoid circular deps if needed, 
#  but here we can import classes for reference)
from perceptron import Perceptron
from adaline import AdalineGD
from mlp import MLP
from visualizer import Visualizer
from network_drawer import NetworkDrawer

class NeuralNetSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Neural Network Learning Simulator")
        self.root.geometry("1600x900")
        
        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # --- Main Layout ---
        # Top Menu
        self.create_menu()
        
        # PanedWindow for 3-column layout
        self.main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left Panel (Controls)
        self.left_panel = ttk.Frame(self.main_pane, width=300)
        self.main_pane.add(self.left_panel, weight=1)
        
        # Center Panel (Visualizations)
        self.center_panel = ttk.Frame(self.main_pane, width=800)
        self.main_pane.add(self.center_panel, weight=4)
        
        # Right Panel (Stats)
        self.right_panel = ttk.Frame(self.main_pane, width=300)
        self.main_pane.add(self.right_panel, weight=1)
        
        # Bottom Panel (Playback Controls - inside center or separate?)
        # Let's put playback controls at the bottom of Center Panel
        
        self.setup_left_panel()
        self.setup_center_panel()
        self.setup_right_panel()
        
        # Controller State
        self.is_playing = False
        self.current_epoch = 0
        self.algorithm = None
        self.dataset = None
        
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
        
        # -- Perceptron Specific --
        # (Could hide/show based on selection, for now just list them)
        
        # -- MLP Specific --
        mlp_labelframe = ttk.LabelFrame(self.left_panel, text="MLP Config")
        mlp_labelframe.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(mlp_labelframe, text="Hidden Layers (e.g., 2,2):").pack(anchor=tk.W)
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
        datasets = ["Linear", "AND", "OR", "XOR", "Circles", "Moons", "Spiral"]
        ttk.OptionMenu(data_labelframe, self.dataset_var, datasets[0], *datasets).pack(fill=tk.X)
        
        ttk.Button(data_labelframe, text="Generate/Load Data", command=self.generate_data).pack(fill=tk.X, pady=5)


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
        
        # Tab 3: Loss Curve
        self.tab_loss = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_loss, text="Metrics")
        
        self.fig_loss = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_loss = self.fig_loss.add_subplot(111)
        self.canvas_loss = FigureCanvasTkAgg(self.fig_loss, self.tab_loss)
        self.canvas_loss.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Playback Controls (Bottom of Center Panel)
        controls_frame = ttk.Frame(self.center_panel)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Start Training", command=self.start_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Step", command=self.step_training).pack(side=tk.LEFT, padx=5)
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
        # (Call data_generator logic here)
        
    def start_training(self):
        self.log_message("Started training...")
        # (Start thread or loop)
        
    def step_training(self):
        self.log_message("Step...")
        
    def reset_simulation(self):
        self.log_message("Simulation Reset.")
        self.epoch_label.config(text="Epoch: 0")
        self.loss_label.config(text="Loss: N/A")
        self.acc_label.config(text="Accuracy: N/A")
