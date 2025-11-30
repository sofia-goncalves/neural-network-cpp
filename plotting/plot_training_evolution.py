#!/usr/bin/env python3
"""
Generate animated GIF showing how (2,16,16,16,1) network evolves during training
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from PIL import Image

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

def create_training_evolution_gif():
    """
    Create animated GIF showing network decision boundary evolution
    """
    # Path to snapshots
    snapshots_dir = config.DATA_OUTPUT / "snapshots"

    # Get all snapshot files
    snapshot_files = sorted(snapshots_dir.glob("snapshot_iter_*.dat"),
                          key=lambda x: int(x.stem.split('_')[-1]))

    if not snapshot_files:
        print(f"No snapshot files found in {snapshots_dir}")
        print("Run ./build/train_with_snapshots first to generate snapshots")
        return

    print(f"Found {len(snapshot_files)} snapshots")

    # Read training data (spiral format has 3 header lines and "end_of_file" footer)
    training_file = config.DATA_INPUT / "spiral_training_data.dat"
    if not training_file.exists():
        print(f"Training data file not found: {training_file}")
        return

    # Read spiral data: skip 3 header lines, use 'end_of_file' as comment marker
    data = np.loadtxt(training_file, skiprows=3, comments='end_of_file')
    inputs = data[:, :2]
    targets = data[:, 2]

    # Create temporary directory for frames
    frames_dir = config.FIGURES_DIR / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Read convergence data
    conv_file = config.CONVERGENCE_LOGS / "snapshot_training_convergence.dat"
    if conv_file.exists():
        conv_data = np.loadtxt(conv_file)
        conv_iterations = conv_data[:, 0]
        conv_cost = conv_data[:, 1]
    else:
        print(f"Warning: Convergence file not found: {conv_file}")
        conv_iterations = None
        conv_cost = None

    # Get snapshot iterations
    snapshot_iterations = [int(f.stem.split('_')[-1]) for f in snapshot_files]

    frame_paths = []

    # Generate a frame for each snapshot
    for idx, snapshot_file in enumerate(snapshot_files):
        # Extract iteration number from filename
        iteration = int(snapshot_file.stem.split('_')[-1])

        print(f"Creating frame {idx+1}/{len(snapshot_files)} (iteration {iteration:,})")

        # Read network output
        network_data = read_network_output(snapshot_file)

        # Create figure with two subplots side by side
        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25)

        # Left: Convergence plot
        ax_conv = fig.add_subplot(gs[0])

        if conv_iterations is not None:
            # Plot full convergence curve
            ax_conv.semilogy(conv_iterations, conv_cost, 'gray', linewidth=1.5, alpha=0.4, label='Training history')

            # Plot markers for all snapshots
            for snap_iter in snapshot_iterations:
                # Find closest iteration in convergence data
                idx_conv = np.argmin(np.abs(conv_iterations - snap_iter))
                if idx_conv < len(conv_cost):
                    ax_conv.semilogy(conv_iterations[idx_conv], conv_cost[idx_conv],
                                   'o', markersize=8, color='lightblue', markeredgecolor='steelblue',
                                   markeredgewidth=1.5, alpha=0.5)

            # Highlight current snapshot with larger, bright marker
            idx_current = np.argmin(np.abs(conv_iterations - iteration))
            if idx_current < len(conv_cost):
                ax_conv.semilogy(conv_iterations[idx_current], conv_cost[idx_current],
                               'o', markersize=16, color='red', markeredgecolor='darkred',
                               markeredgewidth=2.5, label=f'Current (iter {iteration:,})', zorder=10)

        ax_conv.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax_conv.set_ylabel('Cost (log scale)', fontsize=12, fontweight='bold')
        ax_conv.set_title('Training Progress', fontsize=13, fontweight='bold')
        ax_conv.grid(True, alpha=0.3)
        ax_conv.legend(fontsize=10, loc='upper right')

        # Right: Decision boundary
        ax_boundary = fig.add_subplot(gs[1])

        # Plot training data
        plot_training_data(ax_boundary, inputs, targets)

        # Plot decision boundary
        X1, X2, Z = prepare_contour_data(network_data)
        plot_decision_boundary(ax_boundary, X1, X2, Z)

        # Format axes with iteration number
        title = f'Decision Boundary\nIteration {iteration:,}'
        format_axes(ax_boundary, title=title)
        ax_boundary.set_title(title, fontsize=13, fontweight='bold')

        # Save frame
        frame_path = frames_dir / f"frame_{idx:03d}.png"
        plt.tight_layout()
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close()

        frame_paths.append(frame_path)

    # Create GIF from frames
    print("\nCreating animated GIF...")

    frames = [Image.open(fp) for fp in frame_paths]

    # Save as GIF with variable duration
    # First frame (random init): 1.5 seconds
    # Early frames (first 5): 600ms each to see early changes
    # Middle frames: 400ms each
    # Last frame (final): 2 seconds
    durations = [1500]  # First frame
    durations.extend([600] * min(4, len(frames) - 2))  # Early frames
    if len(frames) > 6:
        durations.extend([400] * (len(frames) - 6))  # Middle frames
    durations.append(2000)  # Last frame

    gif_path = config.FIGURES_DIR / "training_evolution.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0  # Loop forever
    )

    print(f"GIF saved to {gif_path}")

    # Also create a static multi-panel figure showing key snapshots
    print("\nCreating static multi-panel figure...")
    create_static_multipanel(snapshot_files, inputs, targets)

    print("\nDone!")

def create_static_multipanel(snapshot_files, inputs, targets):
    """
    Create a static figure showing convergence plot on left and 6 key snapshots on right
    """
    # Specific iterations to display (matching your selection)
    target_iterations = [5000, 50000, 750000, 1000000, 3000000, 6000000]

    # Find indices of snapshots closest to target iterations
    snapshot_iterations = [int(f.stem.split('_')[-1]) for f in snapshot_files]
    indices = []
    selected_iterations = []

    for target_iter in target_iterations:
        # Find closest snapshot to target iteration
        closest_idx = min(range(len(snapshot_iterations)),
                         key=lambda i: abs(snapshot_iterations[i] - target_iter))
        indices.append(closest_idx)
        selected_iterations.append(snapshot_iterations[closest_idx])

    # Read convergence data
    conv_file = config.CONVERGENCE_LOGS / "snapshot_training_convergence.dat"
    if conv_file.exists():
        conv_data = np.loadtxt(conv_file)
        conv_iterations = conv_data[:, 0]
        conv_cost = conv_data[:, 1]
    else:
        conv_iterations = None
        conv_cost = None

    # Create figure with custom layout: convergence plot on left (slightly wider), 2x3 grid on right
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.6, 1, 1, 1], hspace=0.3, wspace=0.3)

    # Left: Convergence plot spanning both rows
    ax_conv = fig.add_subplot(gs[:, 0])

    if conv_iterations is not None:
        # Plot raw noisy data in background at very low alpha
        ax_conv.semilogy(conv_iterations, conv_cost, 'gray', linewidth=0.5, alpha=0.15)

        # Debug: print convergence data range
        print(f"Convergence data range: {conv_iterations.min():.0f} to {conv_iterations.max():.0f}")
        print(f"Selected iterations to plot: {selected_iterations}")

        # Plot markers for selected snapshots with different colors - no outline, lower alpha
        colors = ['red', 'orange', 'gold', 'green', 'blue', 'purple']
        for idx, snap_iter in enumerate(selected_iterations):
            # Find closest iteration in full convergence data
            idx_conv = np.argmin(np.abs(conv_iterations - snap_iter))
            closest_iter = conv_iterations[idx_conv]
            print(f"Marker {idx}: target={snap_iter:,}, closest={closest_iter:.0f}, cost={conv_cost[idx_conv]:.6f}")

            if idx_conv < len(conv_cost):
                ax_conv.semilogy(conv_iterations[idx_conv], conv_cost[idx_conv],
                               'o', markersize=12, color=colors[idx],
                               alpha=0.6,
                               label=f'{snap_iter:,} iter', zorder=10)

    ax_conv.set_xlabel('Iteration', fontsize=13, fontweight='bold')
    ax_conv.set_ylabel('Cost (log scale)', fontsize=13, fontweight='bold')
    ax_conv.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax_conv.grid(True, alpha=0.3)
    ax_conv.legend(fontsize=9, loc='upper right', framealpha=0.9)

    # Right: 2x3 grid of decision boundaries
    colors_for_titles = ['red', 'orange', 'gold', 'green', 'blue', 'purple']
    for panel_idx, snapshot_idx in enumerate(indices):
        snapshot_file = snapshot_files[snapshot_idx]
        iteration = int(snapshot_file.stem.split('_')[-1])

        # Calculate position in grid (2 rows, 3 columns, starting from column 1)
        row = panel_idx // 3
        col = (panel_idx % 3) + 1
        ax = fig.add_subplot(gs[row, col])

        # Read network output
        network_data = read_network_output(snapshot_file)

        # Plot training data
        plot_training_data(ax, inputs, targets)

        # Plot decision boundary
        X1, X2, Z = prepare_contour_data(network_data)
        plot_decision_boundary(ax, X1, X2, Z)

        # Format with colored title matching convergence plot
        title = f'Iteration {iteration:,}'
        format_axes(ax, title=title)
        ax.set_title(title, fontsize=11, fontweight='bold', color=colors_for_titles[panel_idx])

    plt.tight_layout()
    save_path = config.FIGURES_DIR / "training_evolution_multipanel.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Multi-panel figure saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    print("Generating training evolution visualization...")
    print()

    create_training_evolution_gif()
