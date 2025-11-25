#!/usr/bin/env python3
"""
Plot decision boundary for trained neural network
"""

import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from plotting.plot_utils import (
    read_training_data,
    read_network_output,
    prepare_contour_data,
    plot_training_data,
    plot_decision_boundary,
    format_axes
)

def plot_network_decision_boundary(training_file, output_file, save_path):
    """
    Plot training data and decision boundary

    Parameters
    ----------
    training_file : Path
        Path to training data file
    output_file : Path
        Path to network output file
    save_path : Path
        Path to save figure
    """
    # Read data
    inputs, targets = read_training_data(training_file)
    network_data = read_network_output(output_file)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot training data
    plot_training_data(ax, inputs, targets)

    # Plot decision boundary
    X1, X2, Z = prepare_contour_data(network_data)
    plot_decision_boundary(ax, X1, X2, Z)

    # Format axes
    format_axes(ax, title='Neural Network Decision Boundary')

    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    # Example usage
    training_file = config.DATA_OUTPUT / "simple_test_training_data.dat"
    output_file = config.NETWORK_OUTPUTS / "simple_test_output.dat"
    save_path = config.FIGURES_DIR / "simple_test_boundary.png"

    if output_file.exists() and training_file.exists():
        plot_network_decision_boundary(training_file, output_file, save_path)
    else:
        if not training_file.exists():
            print(f"Training data file not found: {training_file}")
        if not output_file.exists():
            print(f"Network output file not found: {output_file}")
        print("Run the network first to generate output data")
