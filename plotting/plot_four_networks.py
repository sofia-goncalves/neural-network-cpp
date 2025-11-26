#!/usr/bin/env python3
"""
Plot comparison of 4 trained networks
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
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

def plot_four_networks_comparison():
    """
    Create a 2x2 grid of subplots showing all 4 trained networks
    """
    # Read training data (same for all networks)
    training_file = config.DATA_OUTPUT / "network_1_training_data.dat"
    inputs, targets = read_training_data(training_file)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for network_id in range(1, 5):
        ax = axes[network_id - 1]

        # Read network output
        output_file = config.NETWORK_OUTPUTS / f"network_{network_id}_output.dat"
        network_data = read_network_output(output_file)

        # Plot training data
        plot_training_data(ax, inputs, targets)

        # Plot decision boundary
        X1, X2, Z = prepare_contour_data(network_data)
        plot_decision_boundary(ax, X1, X2, Z)

        # Format axes
        format_axes(ax, title=f'Network {network_id}')

    # Overall title
    fig.suptitle('Comparison of 4 Trained Networks\nSame Training Data, Different Random Initialisations',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    save_path = config.FIGURES_DIR / "four_networks_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()

def plot_convergence_comparison():
    """
    Plot convergence history for all 4 networks on same axes
    Uses scatter plot to handle fluctuations cleanly
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for network_id in range(1, 5):
        # Read convergence data
        conv_file = config.CONVERGENCE_LOGS / f"network_{network_id}_convergence.dat"
        data = np.loadtxt(conv_file)
        iterations = data[:, 0]
        cost = data[:, 1]

        # Plot with dashed lines for cleaner appearance
        ax.semilogy(iterations, cost, linestyle='--', linewidth=1.5,
                    color=colors[network_id-1], label=f'Network {network_id}', alpha=0.7)

    # Add horizontal line for target tolerance
    ax.axhline(y=1e-4, color='red', linestyle='-', linewidth=2,
               label='Target tolerance (10⁻⁴)', alpha=0.8)

    # Labels and formatting
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cost (log scale)', fontsize=12)
    ax.set_title('Convergence Comparison: 4 Networks', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    # Save
    plt.tight_layout()
    save_path = config.FIGURES_DIR / "four_networks_convergence.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    print("Generating comparison plots for 4 networks...")
    print()

    # Check if all output files exist
    all_files_exist = True
    for network_id in range(1, 5):
        output_file = config.NETWORK_OUTPUTS / f"network_{network_id}_output.dat"
        conv_file = config.CONVERGENCE_LOGS / f"network_{network_id}_convergence.dat"

        if not output_file.exists():
            print(f"Network output file not found: {output_file}")
            all_files_exist = False
        if not conv_file.exists():
            print(f"Convergence file not found: {conv_file}")
            all_files_exist = False

    if not all_files_exist:
        print()
        print("Run ./build/test_four_networks first to generate the data")
    else:
        plot_four_networks_comparison()
        plot_convergence_comparison()
        print()
        print("Done! Check plots/figures/ directory for output")
