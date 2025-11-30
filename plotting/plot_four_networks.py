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

def plot_convergence_only():
    """
    Plot convergence history for all 4 networks overlaid on same axes
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    target_tol = 1e-4

    for network_id in range(1, 5):
        conv_file = config.CONVERGENCE_LOGS / f"network_{network_id}_convergence.dat"

        data = np.loadtxt(conv_file)
        iterations = data[:, 0]
        cost = data[:, 1]

        # Plot raw data to show SGD noise (no moving average)
        ax.semilogy(iterations, cost,
                   linewidth=1.0,
                   color=colors[network_id-1],
                   alpha=0.7,
                   label=f'Network {network_id}')

    # Add horizontal line for target tolerance
    ax.axhline(y=target_tol, color='black', linestyle='--',
              linewidth=1.5, alpha=0.7,
              label=r'Target $\tau = 10^{-4}$')

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Cost (log scale)', fontsize=11)
    ax.set_title('Convergence History', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    save_path = config.FIGURES_DIR / "convergence_only.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Convergence plot saved to {save_path}")
    plt.close()

def plot_boundaries_only():
    """
    Plot decision boundaries in 2x2 grid
    """
    training_file = config.DATA_OUTPUT / "network_1_training_data.dat"
    inputs, targets = read_training_data(training_file)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    for network_id in range(1, 5):
        ax = axes[network_id - 1]

        output_file = config.NETWORK_OUTPUTS / f"network_{network_id}_output.dat"
        network_data = read_network_output(output_file)

        plot_training_data(ax, inputs, targets)

        X1, X2, Z = prepare_contour_data(network_data)
        plot_decision_boundary(ax, X1, X2, Z)

        format_axes(ax, title=f'Network {network_id}')

    plt.tight_layout()
    save_path = config.FIGURES_DIR / "boundaries_only.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Boundaries plot saved to {save_path}")
    plt.close()

def combine_plots_side_by_side():
    """
    Combine convergence and boundaries plots side by side
    """
    from PIL import Image

    conv_path = config.FIGURES_DIR / "convergence_only.png"
    bound_path = config.FIGURES_DIR / "boundaries_only.png"

    # Load images
    conv_img = Image.open(conv_path)
    bound_img = Image.open(bound_path)

    # Resize to same height
    target_height = max(conv_img.height, bound_img.height)

    if conv_img.height != target_height:
        aspect = conv_img.width / conv_img.height
        conv_img = conv_img.resize((int(target_height * aspect), target_height), Image.Resampling.LANCZOS)

    if bound_img.height != target_height:
        aspect = bound_img.width / bound_img.height
        bound_img = bound_img.resize((int(target_height * aspect), target_height), Image.Resampling.LANCZOS)

    # Create combined image
    total_width = conv_img.width + bound_img.width
    combined = Image.new('RGB', (total_width, target_height), 'white')

    combined.paste(conv_img, (0, 0))
    combined.paste(bound_img, (conv_img.width, 0))

    save_path = config.FIGURES_DIR / "four_networks_combined.png"
    combined.save(save_path, dpi=(300, 300))
    print(f"Combined figure saved to {save_path}")

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
        plot_convergence_only()
        plot_boundaries_only()
        combine_plots_side_by_side()
        print()
        print("Done! Check plots/figures/ directory for output")
