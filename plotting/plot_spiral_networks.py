#!/usr/bin/env python3
"""
Plot comparison of different network architectures on spiral data
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Update global font size settings
plt.rcParams.update({'font.size': 16})

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

def plot_network_grid_comparison():
    """
    Create a comprehensive grid plot showing all network configurations
    Rows = depths, Columns = widths (avoids redundancy)
    """
    # Scan for all available spiral network outputs
    all_networks = {}
    for output_file in config.NETWORK_OUTPUTS.glob("spiral_depth_*_width_*_output.dat"):
        # Extract depth and width from filename
        parts = output_file.stem.split('_')
        depth = int(parts[2])
        width = int(parts[4])

        all_networks[(depth, width)] = output_file.stem.replace('_output', '')

    if not all_networks:
        print("No trained networks found for grid comparison")
        return

    # Get unique depths and widths
    depths = sorted(set(d for d, w in all_networks.keys()))
    widths = sorted(set(w for d, w in all_networks.keys()))

    # Read training data (same for all)
    first_network = list(all_networks.values())[0]
    training_file = config.DATA_OUTPUT / f"{first_network}_training_data.dat"
    if not training_file.exists():
        print(f"Training data file not found: {training_file}")
        return
    inputs, targets = read_training_data(training_file)

    # Create grid of subplots: rows=depths, cols=widths
    n_rows = len(depths)
    n_cols = len(widths)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows))

    # Handle case of single row or column
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, depth in enumerate(depths):
        for col_idx, width in enumerate(widths):
            ax = axes[row_idx, col_idx]

            if (depth, width) in all_networks:
                name = all_networks[(depth, width)]
                output_file = config.NETWORK_OUTPUTS / f"{name}_output.dat"

                if output_file.exists():
                    network_data = read_network_output(output_file)

                    # Plot training data
                    plot_training_data(ax, inputs, targets)

                    # Plot decision boundary
                    X1, X2, Z = prepare_contour_data(network_data)
                    plot_decision_boundary(ax, X1, X2, Z)

                    # Generate title with architecture
                    arch = f"({','.join(['2'] + [str(width)] * depth + ['1'])})"
                    title = f"D={depth}, W={width}\n{arch}"
                    format_axes(ax, title=title)
                else:
                    ax.text(0.5, 0.5, f'File not found:\nD={depth}, W={width}',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
            else:
                # Network not trained for this combination
                ax.text(0.5, 0.5, f'Not trained:\nD={depth}, W={width}',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=20, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])

    # Add row and column labels
    for row_idx, depth in enumerate(depths):
        axes[row_idx, 0].set_ylabel(f'Depth {depth}', fontsize=20, fontweight='bold')

    for col_idx, width in enumerate(widths):
        axes[0, col_idx].set_title(f'Width {width}\n' + axes[0, col_idx].get_title(),
                                   fontsize=20, fontweight='bold')
        
    # Save
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    save_path = config.FIGURES_DIR / "spiral_architecture_grid.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Grid comparison saved to {save_path}")
    plt.close()

def plot_convergence_comparison():
    """
    Plot convergence history for all spiral networks
    Line thickness varies dramatically by width for better readability
    """
    # Scan for all available convergence logs and organize by depth and width
    network_configs = {}  # {(depth, width): (name, label, conv_file)}
    for conv_file in sorted(config.CONVERGENCE_LOGS.glob("spiral_depth_*_width_*_convergence.dat")):
        # Extract depth and width from filename
        parts = conv_file.stem.split('_')
        depth = int(parts[2])
        width = int(parts[4])

        name = f"depth_{depth}_width_{width}"
        # Generate architecture string
        arch = f"(2," + ",".join([str(width)] * depth) + ",1)"
        label = f"D={depth}, W={width}"

        network_configs[(depth, width)] = (name, label, conv_file)

    if not network_configs:
        print("No convergence logs found")
        return

    # Get unique depths and widths
    depths = sorted(set(d for d, w in network_configs.keys()))
    widths = sorted(set(w for d, w in network_configs.keys()))

    # Color map: depth determines base color
    depth_colors = {2: '#1f77b4', 3: '#2ca02c', 4: '#d62728'}  # blue, green, red

    fig, ax = plt.subplots(figsize=(12, 7))

    for (depth, width), (name, label, conv_file) in sorted(network_configs.items()):
        data = np.loadtxt(conv_file)
        iterations = data[:, 0]
        cost = data[:, 1]

        # Apply moving average smoothing
        window_size = max(1, len(cost) // 50)
        if window_size > 1:
            cost_smooth = np.convolve(cost, np.ones(window_size)/window_size, mode='valid')
            iterations_smooth = iterations[window_size-1:]
        else:
            cost_smooth = cost
            iterations_smooth = iterations

        # Color based on depth
        base_color = depth_colors.get(depth, 'gray')

        # Line thickness based on width - make differences STARK
        # Width 4: thin (0.8), Width 8: medium (2.0), Width 16: thick (3.5)
        width_linewidths = {4: 0.8, 8: 1.5, 16: 4.5}
        width_alpha = {4: 1.0, 8: 0.8, 16: 0.3}
        linewidth = width_linewidths.get(width, 1.5)
        alpha = width_alpha.get(width, 0.8)

        # Plot
        ax.semilogy(iterations_smooth, cost_smooth, linestyle='-', linewidth=linewidth,
                    color=base_color, label=label, alpha=alpha)

    # Labels and formatting
    ax.set_xlabel('Iteration', fontsize=16)
    ax.set_ylabel('Cost (log scale)', fontsize=16)
    # ax.set_title('Convergence Comparison: Spiral Networks', fontsize=16, fontweight='bold')
    ax.legend(fontsize=14, loc='best', ncol=1)
    ax.grid(True, alpha=0.3)

    # Save
    plt.tight_layout()
    save_path = config.FIGURES_DIR / "spiral_convergence_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Convergence comparison saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    print("Generating comparison plots for spiral networks...")
    print()

    plot_network_grid_comparison()
    plot_convergence_comparison()

    print()
    print("Done! Check plots/figures/ directory for output")