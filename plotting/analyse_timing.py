#!/usr/bin/env python3
"""
Analyse results from computational cost test
Fits experimental data to N^a_layer × N^b_neuron model
Compares fitted exponents with theoretical predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def load_timing_data():
    """Load timing results from data file"""
    results_file = config.DATA_OUTPUT / "timing_results.dat"

    if not results_file.exists():
        print(f"Error: {results_file} not found")
        print("Run ./build/timing_analysis first")
        return None

    # Load data: n_layer n_neuron n_params time_feedforward time_fd time_backprop ratio_fd_to_ff ratio_bp_to_ff ratio_fd_to_bp
    data = np.loadtxt(results_file)

    return data

def fit_power_law(data, algorithm_index):
    """
    Fit data to power law model: time = C × N_layer^a × N_neuron^b

    Parameters
    ----------
    data : ndarray
        Timing data from load_timing_data()
    algorithm_index : int
        Column index for algorithm timing (3=FF, 4=FD, 5=BP)

    Returns
    -------
    a, b, C : float
        Fitted exponents and coefficient
    """
    n_layer = data[:, 0]
    n_neuron = data[:, 1]
    time = data[:, algorithm_index]

    # Fit log-linear model: log(time) = log(C) + a*log(N_layer) + b*log(N_neuron)
    # Using least squares

    X = np.column_stack([np.ones_like(n_layer), np.log(n_layer), np.log(n_neuron)])
    y = np.log(time)

    # Solve normal equations: X^T X θ = X^T y
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

    log_C, a, b = coeffs
    C = np.exp(log_C)

    return a, b, C

def fit_univariate_b(data, algorithm_index):
    """
    Fit b exponent by fixing N_layer and varying N_neuron
    Returns mean and std of b across all N_layer slices
    """
    n_layer = data[:, 0]
    n_neuron = data[:, 1]
    time = data[:, algorithm_index]

    unique_layers = np.unique(n_layer)
    b_values = []

    for n_lay in unique_layers:
        mask = (n_layer == n_lay)
        if np.sum(mask) < 3:  # Need at least 3 points
            continue

        neurons = n_neuron[mask]
        t = time[mask]

        # Fit log(t) = alpha + b*log(neurons)
        X = np.column_stack([np.ones_like(neurons), np.log(neurons)])
        y = np.log(t)
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        b_values.append(coeffs[1])

    return np.mean(b_values), np.std(b_values)

def fit_univariate_a(data, algorithm_index):
    """
    Fit a exponent by fixing N_neuron and varying N_layer
    Returns mean and std of a across all N_neuron slices
    """
    n_layer = data[:, 0]
    n_neuron = data[:, 1]
    time = data[:, algorithm_index]

    unique_neurons = np.unique(n_neuron)
    a_values = []

    for n_neur in unique_neurons:
        mask = (n_neuron == n_neur)
        if np.sum(mask) < 3:  # Need at least 3 points
            continue

        layers = n_layer[mask]
        t = time[mask]

        # Fit log(t) = alpha + a*log(layers)
        X = np.column_stack([np.ones_like(layers), np.log(layers)])
        y = np.log(t)
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        a_values.append(coeffs[1])

    return np.mean(a_values), np.std(a_values)

def compute_r_squared(data, algorithm_index, a, b, C):
    """Compute R^2 for the fit"""
    n_layer = data[:, 0]
    n_neuron = data[:, 1]
    time_measured = data[:, algorithm_index]

    time_predicted = C * (n_layer ** a) * (n_neuron ** b)

    ss_res = np.sum((time_measured - time_predicted) ** 2)
    ss_tot = np.sum((time_measured - np.mean(time_measured)) ** 2)

    r_squared = 1 - (ss_res / ss_tot)

    return r_squared

def print_fit_results():
    """Print fitting results and comparison with theory"""
    data = load_timing_data()

    if data is None:
        return

    print("\nExercise 3: Computational Cost Analysis -- Experimental Results\n")

    # Theoretical predictions
    theory = {
        'Feed-forward': (1.0, 2.0),
        'Finite-diff': (2.0, 4.0),
        'Back-prop': (1.0, 2.0)
    }

    algorithm_names = ['Feed-forward', 'Finite-diff', 'Back-prop']
    algorithm_indices = [3, 4, 5]

    print("=" * 95)
    print("MULTIVARIATE FIT: log(T) = log(C) + a*log(N_layer) + b*log(N_neuron)")
    print("=" * 95)
    print(f"{'Algorithm':<20} {'Theoretical':<20} {'Experimental':<25} {'R²':<10}")
    print(f"{'':20} {'(a, b)':<20} {'(a, b)':<25}")
    print("-" * 95)

    for name, idx in zip(algorithm_names, algorithm_indices):
        # Fit experimental data
        a_exp, b_exp, C = fit_power_law(data, idx)

        # Compute R^2
        r2 = compute_r_squared(data, idx, a_exp, b_exp, C)

        # Theoretical values
        a_theory, b_theory = theory[name]

        print(f"{name:<20} ({a_theory:.1f}, {b_theory:.1f}){'':<12} "
              f"({a_exp:.3f}, {b_exp:.3f}){'':<10} "
              f"{r2:.4f}")

    print("\n")
    print("=" * 100)
    print("UNIVARIATE FITS: Isolating each variable")
    print("=" * 100)
    print(f"{'Algorithm':<20} {'Theory':<15} {'Univariate a':<25} {'Univariate b':<25}")
    print(f"{'':20} {'(a, b)':<15} {'(mean ± std)':<25} {'(mean ± std)':<25}")
    print("-" * 100)

    for name, idx in zip(algorithm_names, algorithm_indices):
        a_theory, b_theory = theory[name]

        # Univariate fits
        a_mean, a_std = fit_univariate_a(data, idx)
        b_mean, b_std = fit_univariate_b(data, idx)

        print(f"{name:<20} ({a_theory:.1f}, {b_theory:.1f}){'':<7} "
              f"({a_mean:.2f} ± {a_std:.2f}){'':<12} "
              f"({b_mean:.2f} ± {b_std:.2f})")

    print("\n")

def plot_scaling_analysis():
    """Generate scaling plots for all three algorithms"""
    data = load_timing_data()

    if data is None:
        return

    n_layer = data[:, 0]
    n_neuron = data[:, 1]

    algorithm_names = ['Feed-forward', 'Finite-differencing', 'Back-propagation']
    algorithm_indices = [3, 4, 5]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, name, idx in zip(axes, algorithm_names, algorithm_indices):
        time = data[:, idx]

        # Fit power law
        a_exp, b_exp, C = fit_power_law(data, idx)
        r2 = compute_r_squared(data, idx, a_exp, b_exp, C)

        # Plot measured vs predicted (log-log)
        time_predicted = C * (n_layer ** a_exp) * (n_neuron ** b_exp)

        ax.loglog(time, time_predicted, 'o', markersize=6, alpha=0.6, label='Data')

        # Plot perfect fit line
        min_time, max_time = time.min(), time.max()
        ax.loglog([min_time, max_time], [min_time, max_time], 'k--',
                  linewidth=1.5, label='Perfect fit')

        ax.set_xlabel('Measured time (s)', fontsize=11)
        ax.set_ylabel('Predicted time (s)', fontsize=11)
        ax.set_title(f'{name}\n$a = {a_exp:.2f}$, $b = {b_exp:.2f}$, $R^2 = {r2:.4f}$',
                     fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = config.FIGURES_DIR / "timing_scaling_fits.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Scaling fits plot saved to: {output_file}")

    plt.close()

def plot_algorithm_comparison():
    """Plot comparison of three algorithms' timings"""
    data = load_timing_data()

    if data is None:
        return

    n_layer = data[:, 0]
    n_neuron = data[:, 1]

    time_ff = data[:, 3]
    time_fd = data[:, 4]
    time_bp = data[:, 5]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Absolute timings
    ax = axes[0]

    # Group by N_layer for visualization
    unique_layers = np.unique(n_layer)

    for n_lay in unique_layers:
        mask = (n_layer == n_lay)
        neurons = n_neuron[mask]

        ax.loglog(neurons, time_ff[mask], 'o-', label=f'FF, L={int(n_lay)}',
                  markersize=5, alpha=0.7)
        ax.loglog(neurons, time_bp[mask], 's-', label=f'BP, L={int(n_lay)}',
                  markersize=5, alpha=0.7)
        ax.loglog(neurons, time_fd[mask], '^-', label=f'FD, L={int(n_lay)}',
                  markersize=5, alpha=0.7)

    ax.set_xlabel('$N_{\\text{neuron}}$', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('Absolute Computation Time', fontsize=13)
    ax.legend(fontsize=7, ncol=3, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 2: Ratios (FD/BP and BP/FF)
    ax = axes[1]

    ratio_fd_to_bp = time_fd / time_bp
    ratio_bp_to_ff = time_bp / time_ff

    for n_lay in unique_layers:
        mask = (n_layer == n_lay)
        neurons = n_neuron[mask]

        ax.semilogy(neurons, ratio_fd_to_bp[mask], 'o-',
                    label=f'FD/BP, L={int(n_lay)}', markersize=5, alpha=0.7)
        ax.semilogy(neurons, ratio_bp_to_ff[mask], 's-',
                    label=f'BP/FF, L={int(n_lay)}', markersize=5, alpha=0.7)

    ax.set_xlabel('$N_{\\text{neuron}}$', fontsize=12)
    ax.set_ylabel('Time ratio', fontsize=12)
    ax.set_title('Algorithm Efficiency Comparison', fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = config.FIGURES_DIR / "timing_algorithm_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Algorithm comparison plot saved to: {output_file}")

    plt.close()

def generate_latex_table():
    """Generate LaTeX table of fit results"""
    data = load_timing_data()

    if data is None:
        return

    algorithm_names = ['Feed-forward', 'Finite-differencing', 'Back-propagation']
    algorithm_indices = [3, 4, 5]

    theory = {
        'Feed-forward': (1.0, 2.0),
        'Finite-differencing': (2.0, 4.0),
        'Back-propagation': (1.0, 2.0)
    }

    latex_lines = []
    latex_lines.append("% Computational cost analysis results")
    latex_lines.append("% Generated by analyze_timing.py")
    latex_lines.append("\\begin{table}")
    latex_lines.append("\\footnotesize")
    latex_lines.append("\\begin{tabular}{lccc}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Algorithm} & \\textbf{Theory} & \\textbf{Experimental} & \\textbf{$R^2$} \\\\")
    latex_lines.append("& $(a, b)$ & $(a, b)$ & \\\\")
    latex_lines.append("\\hline")

    for name, idx in zip(algorithm_names, algorithm_indices):
        a_exp, b_exp, C = fit_power_law(data, idx)
        r2 = compute_r_squared(data, idx, a_exp, b_exp, C)
        a_theory, b_theory = theory[name]

        latex_lines.append(f"{name} & $({a_theory:.0f}, {b_theory:.0f})$ & "
                          f"$({a_exp:.2f}, {b_exp:.2f})$ & {r2:.3f} \\\\")

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    output_file = config.DATA_OUTPUT / "timing_analysis_table.tex"
    with open(output_file, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"LaTeX table saved to: {output_file}")

if __name__ == "__main__":
    print("Analysing timing results from Exercise 3\n")

    # Print fit results to console
    print_fit_results()

    # Generate plots
    plot_scaling_analysis()
    plot_algorithm_comparison()

    # Generate LaTeX table
    generate_latex_table()

    print("\nAnalysis complete")
