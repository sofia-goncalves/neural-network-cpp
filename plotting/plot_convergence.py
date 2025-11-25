#!/usr/bin/env python3
"""
Plot convergence history for neural network training
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def plot_convergence(convergence_file, save_path, semilogy=True):
    """
    Plot cost vs iteration

    Parameters
    ----------
    convergence_file : Path
        Path to convergence log file (format: iteration cost)
    save_path : Path
        Path to save figure
    semilogy : bool
        Use log scale for y-axis
    """
    # Read data
    data = np.loadtxt(convergence_file)
    iterations = data[:, 0]
    cost = data[:, 1]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    if semilogy:
        ax.semilogy(iterations, cost, linewidth=2)
        ax.set_ylabel('Cost (log scale)')
    else:
        ax.plot(iterations, cost, linewidth=2)
        ax.set_ylabel('Cost')

    ax.set_xlabel('Iteration')
    ax.set_title('Training Convergence History')
    ax.grid(True, alpha=0.3)

    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    # Example usage
    convergence_file = config.CONVERGENCE_LOGS / "simple_test_convergence.dat"
    save_path = config.FIGURES_DIR / "simple_test_convergence.png"

    if convergence_file.exists():
        plot_convergence(convergence_file, save_path)
    else:
        print(f"Convergence file not found: {convergence_file}")
        print("Run training first to generate convergence data")
