"""
Main entry point for the Neural Network Learning Simulator.
Initializes the root window and starts the Graphical User Interface.
"""
import customtkinter as ctk
from gui import NeuralNetSimulatorGUI

def main():
    """Application initialization and main loop."""
    root = ctk.CTk()
    # Let GUI handle its own styling/appearance internally to be consistent
    app = NeuralNetSimulatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
