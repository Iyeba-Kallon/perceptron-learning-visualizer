import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
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
    """
    The main Graphical User Interface for the Neural Network Simulator.
    Manages the layout, user interactions, and the main training event loop.
    """
    def __init__(self, root):
        """
        Initialize the GUI and set up the main application state.

        Args:
            root (customtkinter.CTk): The root window object.
        """
        self.root = root
        self.root.title("Advanced Neural Network Learning Simulator")
        self.root.geometry("1600x900")
        
        # Configure Look and Feel
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # --- Main Layout Architecture ---
        self._initialize_layout()
        self.create_menu()
        self._setup_panels()
        
        # --- Application Control State ---
        self.is_playing = False
        self.current_epoch = 0
        self.algorithm = None
        self.X = None
        self.y = None
        self.history = {'loss': [], 'accuracy': []}
        
        # --- Visualization Engines ---
        # Note: Axes are created in setup_center_panel
        self.vis = Visualizer(self.ax_boundary)
        self.net_drawer = NetworkDrawer(self.ax_network)
        
        # --- Event Bindings ---
        # Connect plot interactions (for Manual Dataset mode)
        self.canvas_boundary.mpl_connect('button_press_event', self.on_click)

    def _initialize_layout(self):
        """Configure the grid weights for the main window resize behavior."""
        self.root.grid_columnconfigure(0, weight=1) # Left Sidebar
        self.root.grid_columnconfigure(1, weight=4) # Center Plot Area
        self.root.grid_columnconfigure(2, weight=1) # Right Event Log
        self.root.grid_rowconfigure(0, weight=1)

    def _setup_panels(self):
        """Instantiate the main containers for each UI section."""
        self.left_panel = ctk.CTkScrollableFrame(self.root, width=300, corner_radius=10)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.center_panel = ctk.CTkFrame(self.root, corner_radius=10)
        self.center_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.right_panel = ctk.CTkFrame(self.root, width=300, corner_radius=10)
        self.right_panel.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        
        # Initialize sub-component widgets
        self.setup_left_panel()
        self.setup_center_panel()
        self.setup_right_panel()

    def on_click(self, event):
        """
        Handle mouse click events on the decision boundary plot.
        Used to manually add data points when "Manual" mode is selected.
        
        Args:
            event (matplotlib.backend_bases.MouseEvent): The click event.
        """
        if event.inaxes != self.ax_boundary or self.dataset_var.get() != "Manual":
            return
            
        # Left click (1) -> Class 0, Right click (3) -> Class 1
        label = 0 if event.button == 1 else 1
        
        x, y = event.xdata, event.ydata
        if x is None or y is None: 
            return
        
        # Append to dataset
        new_point = np.array([[x, y]])
        new_label = np.array([label])
        
        if self.X is None:
            self.X, self.y = new_point, new_label
        else:
            self.X = np.vstack([self.X, new_point])
            self.y = np.hstack([self.y, new_label])
            
        self.log_message(f"Point added: ({x:.2f}, {y:.2f}) -> Class {label}")
        self._repaint_manual_data()

    def _repaint_manual_data(self):
        """Redraw the manually entered data points on the plot."""
        self.ax_boundary.clear()
        markers = ('o', 's')
        colors = ('#ff3333', '#33adff')
        
        for idx, cl in enumerate(np.unique(self.y)):
            c_idx = int(cl) % len(colors)
            m_idx = int(cl) % len(markers)
            
            mask = (self.y == cl)
            self.ax_boundary.scatter(x=self.X[mask, 0], 
                                     y=self.X[mask, 1],
                                     alpha=0.9, 
                                     c=colors[c_idx],
                                     marker=markers[m_idx], 
                                     label=f'Class {cl}', 
                                     edgecolor='white',
                                     s=60)
        
        # Set viewport padding around points
        margin = 1.0
        self.ax_boundary.set_xlim(min(self.X[:,0]) - margin, max(self.X[:,0]) + margin)
        self.ax_boundary.set_ylim(min(self.X[:,1]) - margin, max(self.X[:,1]) + margin)
        self.ax_boundary.tick_params(colors='white')
        
        self._style_plot_legend(self.ax_boundary)
        self.canvas_boundary.draw()

    def _style_plot_legend(self, ax):
        """Apply a professional dark theme to the plot legend."""
        legend = ax.legend(loc='upper left', frameon=True)
        frame = legend.get_frame()
        frame.set_facecolor('#2b2b2b')
        frame.set_edgecolor('#555555')
        for text in legend.get_texts():
            text.set_color("white")
        
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
        """Construct the sidebar for model selection and parameter configuration."""
        # 1. Algorithm Selection
        self._add_section_header(self.left_panel, "Algorithm Selection")
        self.algo_var = ctk.StringVar(value="Perceptron")
        for algo in ["Perceptron", "Adaline", "MLP"]:
            ctk.CTkRadioButton(self.left_panel, text=algo, variable=self.algo_var, value=algo).pack(anchor=tk.W, pady=2)
        
        # 2. Universal Hyperparameters
        self._add_section_header(self.left_panel, "Hyperparameters", pady=(15, 10))
        
        ctk.CTkLabel(self.left_panel, text="Learning Rate (η):").pack(anchor=tk.W)
        self.lr_scale = ctk.CTkSlider(self.left_panel, from_=0.0001, to=1.0, number_of_steps=1000)
        self.lr_scale.set(0.01)
        self.lr_scale.pack(fill=tk.X, pady=(0, 10))
        
        ctk.CTkLabel(self.left_panel, text="Epochs:").pack(anchor=tk.W)
        self.epochs_entry = ctk.CTkEntry(self.left_panel)
        self.epochs_entry.insert(0, "50")
        self.epochs_entry.pack(fill=tk.X, pady=(0, 10))
        
        # 3. MLP-Specific Configuration
        self._add_section_header(self.left_panel, "MLP Config", pady=(15, 10))
        
        ctk.CTkLabel(self.left_panel, text="Hidden Layers (e.g., 4):").pack(anchor=tk.W)
        self.hidden_layers_entry = ctk.CTkEntry(self.left_panel)
        self.hidden_layers_entry.insert(0, "4")
        self.hidden_layers_entry.pack(fill=tk.X, pady=(0, 10))
        
        ctk.CTkLabel(self.left_panel, text="Activation:").pack(anchor=tk.W)
        self.activation_var = ctk.StringVar(value="Sigmoid")
        activations = ["Sigmoid", "Tanh", "ReLU", "Hardlim", "Hardlims"]
        ctk.CTkOptionMenu(self.left_panel, variable=self.activation_var, values=activations).pack(fill=tk.X, pady=(0, 10))

        # 4. Dataset Selection
        self._add_section_header(self.left_panel, "Dataset", pady=(15, 10))
        self.dataset_var = ctk.StringVar(value="Linear")
        datasets = ["Linear", "AND", "OR", "XOR", "Circles", "Moons", "Spiral", "Manual", "Custom (CSV/TXT)"]
        ctk.CTkOptionMenu(self.left_panel, variable=self.dataset_var, values=datasets).pack(fill=tk.X, pady=(0, 10))
        
        ctk.CTkButton(self.left_panel, text="Generate/Load Data", command=self.generate_data).pack(fill=tk.X, pady=10)
        ctk.CTkLabel(self.left_panel, text="(Manual: L-Click=0, R-Click=1)", font=("Arial", 10)).pack(anchor=tk.W)

    def _add_section_header(self, parent, text, pady=(5, 10)):
        """Utility to add a bold section header to a panel."""
        header = ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=16, weight="bold"))
        header.pack(fill=tk.X, padx=5, pady=pady)


    def setup_center_panel(self):
        """Set up the primary visualization area with tabs and playback controls."""
        self.notebook = ctk.CTkTabview(self.center_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- Tab 1: Decision Boundary ---
        self.tab_boundary = self.notebook.add("Decision Boundary")
        self.fig_boundary = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_boundary = self.fig_boundary.add_subplot(111)
        self._set_matplotlib_bg(self.fig_boundary, self.ax_boundary)
        self.canvas_boundary = FigureCanvasTkAgg(self.fig_boundary, self.tab_boundary)
        self.canvas_boundary.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # --- Tab 2: Network Architecture ---
        self.tab_network = self.notebook.add("Network Architecture")
        self.fig_network = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_network = self.fig_network.add_subplot(111)
        self._set_matplotlib_bg(self.fig_network, self.ax_network)
        self.canvas_network = FigureCanvasTkAgg(self.fig_network, self.tab_network)
        self.canvas_network.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # --- Tab 3: Training Metrics ---
        self.tab_loss = self.notebook.add("Metrics")
        self.fig_loss = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_loss = self.fig_loss.add_subplot(111)
        self._set_matplotlib_bg(self.fig_loss, self.ax_loss)
        self.canvas_loss = FigureCanvasTkAgg(self.fig_loss, self.tab_loss)
        self.canvas_loss.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # --- Playback Control Bar ---
        self._setup_playback_controls()

    def _set_matplotlib_bg(self, fig, ax):
        """Set a consistent dark background for a Matplotlib figure/axes."""
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')

    def _setup_playback_controls(self):
        """Add training and reset buttons to the center panel."""
        frame = ctk.CTkFrame(self.center_panel, fg_color="transparent")
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.btn_start = ctk.CTkButton(frame, text="Start Training", command=self.start_training, 
                                       fg_color="#2ecc71", hover_color="#27ae60", text_color="white")
        self.btn_start.pack(side=tk.LEFT, padx=10)
        
        self.btn_stop = ctk.CTkButton(frame, text="Stop", command=self.stop_training, 
                                      state=tk.DISABLED, fg_color="#e74c3c", hover_color="#c0392b", text_color="white")
        self.btn_stop.pack(side=tk.LEFT, padx=10)
        
        ctk.CTkButton(frame, text="Reset", command=self.reset_simulation, 
                      fg_color="#f39c12", hover_color="#d68910", text_color="white").pack(side=tk.LEFT, padx=10)

    def setup_right_panel(self):
        """Create the dashboard for training statistics and system logs."""
        self._add_section_header(self.right_panel, "Training Stats", pady=(10, 5))
        
        self.epoch_label = ctk.CTkLabel(self.right_panel, text="Epoch: 0", font=ctk.CTkFont(size=14))
        self.epoch_label.pack(anchor=tk.W, padx=10, pady=2)
        
        self.loss_label = ctk.CTkLabel(self.right_panel, text="Loss: N/A", font=ctk.CTkFont(size=14))
        self.loss_label.pack(anchor=tk.W, padx=10, pady=2)
        
        self.acc_label = ctk.CTkLabel(self.right_panel, text="Accuracy: N/A", font=ctk.CTkFont(size=14))
        self.acc_label.pack(anchor=tk.W, padx=10, pady=2)
        
        self._add_section_header(self.right_panel, "Event Log", pady=(20, 5))
        self.log_text = ctk.CTkTextbox(self.right_panel, height=200, state='disabled', corner_radius=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def log_message(self, msg):
        """Thread-safe logging of messages to the application's event log."""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, f"{msg}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')

    def show_about(self):
        messagebox.showinfo("About", "Neural Network Learning Simulator v1.0")
            def generate_data(self):
        """Generate or load the dataset based on current user selection."""
        selection = self.dataset_var.get()
        self.log_message(f"Loading/Generating: {selection}")
        
        # Reset state
        self.X, self.y = None, None

        if selection == "Manual":
             self._handle_manual_mode_init()
             return

        if selection == "Custom (CSV/TXT)":
             if not self._handle_custom_file_load():
                 return
        else:
            # Generate synthetic datasets using the generator module
            self._generate_synthetic_data(selection)
        
        if self.X is not None:
            self._render_initial_plot()
            self.log_message(f"Dataset ready. Samples: {len(self.y)}")

    def _handle_manual_mode_init(self):
        """Initialize the manual drawing mode."""
        self.log_message("Manual mode: Click the plot to add data points.")
        self.ax_boundary.clear()
        self.ax_boundary.tick_params(colors='white')
        self.canvas_boundary.draw()

    def _handle_custom_file_load(self):
        """Open a file dialog to parse a custom CSV or text dataset."""
        self.log_message("Requesting file path...")
        filepath = ctk.filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("Text/CSV files", "*.csv *.txt"), ("All files", "*.*")]
        )
        if not filepath:
            self.log_message("Operation cancelled by user.")
            return False
        
        try:
            # Load numeric data (comma or space separated)
            data = np.loadtxt(filepath, delimiter=',' if filepath.endswith('.csv') else None)
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError("Data format invalid. Requirements: At least 2 columns [1+ Features, 1 Label].")
            
            # Assume last column is target label, others are features
            self.X, self.y = data[:, :-1], data[:, -1]
            self.log_message(f"File loaded: {filepath}")
            return True
        except Exception as e:
            self.log_message(f"Error: {e}")
            messagebox.showerror("IO Error", f"Could not load dataset:\n{e}")
            return False

    def _generate_synthetic_data(self, selection):
        """Map dataset names to generator functions."""
        generators = {
            "Linear": data_generator.generate_linear_data,
            "Circles": data_generator.generate_circles_data,
            "Moons": data_generator.generate_moons_data,
            "Spiral": data_generator.generate_spiral_data,
            "AND": lambda: logic_gate_data("AND"),
            "OR": lambda: logic_gate_data("OR"),
            "XOR": lambda: logic_gate_data("XOR")
        }
        if selection in generators:
            self.X, self.y = generators[selection]()

    def _render_initial_plot(self):
        """Helper to draw the raw data points before training begins."""
        self.ax_boundary.clear()
        markers = ('o', 's')
        colors = ('#ff3333', '#33adff')
        
        for idx, label in enumerate(np.unique(self.y)):
            mask = (self.y == label)
            self.ax_boundary.scatter(x=self.X[mask, 0], 
                                     y=self.X[mask, 1],
                                     alpha=0.9, 
                                     c=colors[idx % len(colors)],
                                     marker=markers[idx % len(markers)], 
                                     label=f'Class {label}', 
                                     edgecolor='white',
                                     s=60)
        self.ax_boundary.tick_params(colors='white')
        self._style_plot_legend(self.ax_boundary)
        self.canvas_boundary.draw()

    def init_algorithm(self):
        """
        Instantiate the selected machine learning model with user-provided parameters.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        algo_name = self.algo_var.get()
        lr = self.lr_scale.get()
        
        try:
            epochs = int(self.epochs_entry.get())
        except ValueError:
            messagebox.showerror("Validation Error", "Epochs must be an integer sequence.")
            return False
        
        # Dispatch model creation
        if algo_name == "Perceptron":
            self.algorithm = Perceptron(eta=lr, n_iter=epochs)
        elif algo_name == "Adaline":
            self.algorithm = AdalineGD(eta=lr, n_iter=epochs)
        elif algo_name == "MLP":
            if not self._init_mlp(lr, epochs):
                return False
                
        return True

    def _init_mlp(self, lr, epochs):
        """Parse hidden layer configuration and initialize the MLP."""
        hidden_str = self.hidden_layers_entry.get()
        try:
            # Parse CSV string (e.g., "4, 4") into integer list
            hidden = [int(x.strip()) for x in hidden_str.split(',') if x.strip()]
        except ValueError:
            messagebox.showerror("Config Error", "Hidden Layers must be comma-separated integers (e.g. '4,4').")
            return False
            
        # Architecture: [Input] + [Hidden...] + [Output]
        layer_config = [self.X.shape[1]] + hidden + [1]
        activation = self.activation_var.get().lower()
        
        self.algorithm = MLP(layers=layer_config, activation=activation, eta=lr, epochs=epochs)
        
        # Render initial weight visualization
        self.net_drawer.draw(layer_config, self.algorithm.weights)
        self.canvas_network.draw()
        return True

    def start_training(self):
        """Begin the asynchronous training process."""
        if self.X is None:
            messagebox.showwarning("Incomplete Data", "Please load or generate a dataset first.")
            return
            
        if self.is_playing:
            return

        if not self.init_algorithm():
            return
            
        # Update UI state
        self.is_playing = True
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.current_epoch = 0
        self.history = {'loss': [], 'accuracy': []}
        
        self.log_message(f"Initiating training session: {self.algo_var.get()}")
        self.train_loop()
        
    def stop_training(self):
        """Halt the training loop."""
        self.is_playing = False
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.log_message("Training interrupted.")

    def reset_simulation(self):
        """Restore the simulation to its initial state."""
        self.stop_training()
        self.algorithm = None
        self.current_epoch = 0
        self.history = {'loss': [], 'accuracy': []}
        
        # Clear all visualizations
        self.ax_boundary.clear()
        self.ax_network.clear()
        self.ax_loss.clear()
        self.canvas_boundary.draw()
        self.canvas_network.draw()
        self.canvas_loss.draw()
        
        # Reset labels
        self.epoch_label.configure(text="Epoch: 0")
        self.loss_label.configure(text="Loss: N/A")
        self.acc_label.configure(text="Accuracy: N/A")
        
        self.log_message("System reset complete.")
        
        # Restore data if not in manual mode
        if self.X is not None and self.dataset_var.get() != "Manual":
             self._render_initial_plot() 
        elif self.dataset_var.get() == "Manual":
             self.X, self.y = None, None
    def train_loop(self):
        """
        The main training event loop.
        Executes one epoch, updates visualizations, and schedules the next step.
        """
        if not self.is_playing:
            return
            
        max_epochs = int(self.epochs_entry.get())
        if self.current_epoch >= max_epochs:
            self.stop_training()
            self.log_message("Convergence limit reached.")
            return
            
        # --- Perform Training Step ---
        loss, acc = self._execute_epoch_step()

        self.current_epoch += 1
        self.history['loss'].append(loss)
        self.history['accuracy'].append(acc)
        
        # --- UI Dashboard Update ---
        self.epoch_label.configure(text=f"Epoch: {self.current_epoch}/{max_epochs}")
        self.loss_label.configure(text=f"Loss: {loss:.4f}")
        self.acc_label.configure(text=f"Accuracy: {acc*100:.1f}%")
        
        # --- Visualization Updates ---
        # 1. Decision Boundary
        self.vis.plot_decision_regions(self.X, self.y, self.algorithm)
        self.canvas_boundary.draw()
        
        # 2. Performance Metrics
        self.vis.plot_metrics(self.ax_loss, self.history, metric='loss')
        self.canvas_loss.draw()
        
        # Schedule next iteration via Tkinter's event loop
        # We use a small delay for animation visibility
        self.root.after(10, self.train_loop)

    def _execute_epoch_step(self):
        """Dispatch training step logic based on the active algorithm."""
        loss, acc = 0, 0
        
        if isinstance(self.algorithm, MLP):
            loss, _ = self.algorithm.train_step(self.X, self.y)
            preds = self.algorithm.predict(self.X)
            acc = accuracy(self.y.reshape(-1, 1), preds.reshape(-1, 1))
            
            # Sync Architecture Diagram with new weight strengths
            self.net_drawer.draw(self.algorithm.layers, self.algorithm.weights)
            self.canvas_network.draw()
            
        elif isinstance(self.algorithm, (AdalineGD, Perceptron)):
             # Single epoch pass
             if isinstance(self.algorithm, AdalineGD):
                 loss, _ = self.algorithm.train_step(self.X, self.y)
             else:
                 # Perceptron Online Learning (Stochastic)
                 total_errors = 0
                 indices = np.random.permutation(len(self.y))
                 for idx in indices:
                     step_error, _ = self.algorithm.train_step(self.X[idx], self.y[idx])
                     total_errors += step_error
                 loss = total_errors
             
             preds = self.algorithm.predict(self.X)
             acc = accuracy(self.y, preds)
             
        return loss, acc
