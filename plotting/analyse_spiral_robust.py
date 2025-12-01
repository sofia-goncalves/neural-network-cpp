#!/usr/bin/env python3
"""
Computes mean, std dev, and other statistics for each architecture
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def load_robust_results():
    """Load robust results from data file"""
    results_file = config.DATA_OUTPUT / "spiral_robust_results.dat"

    if not results_file.exists():
        print(f"Error: {results_file} not found")
        print("Run ./build/test_spiral_networks_robust first")
        return None

    # Load data: depth width repeat final_cost converged iterations improvement_rate
    data = np.loadtxt(results_file)

    return data

def compute_statistics(data):
    """
    Compute statistics for each architecture (depth, width) combination

    Returns dictionary with keys (depth, width) and values containing statistics
    """
    if data is None:
        return None

    stats = {}

    # Get unique (depth, width) combinations
    depths = np.unique(data[:, 0]).astype(int)
    widths = np.unique(data[:, 1]).astype(int)

    for depth in depths:
        for width in widths:
            # Filter data for this architecture
            mask = (data[:, 0] == depth) & (data[:, 1] == width)
            arch_data = data[mask]

            if len(arch_data) == 0:
                continue

            final_costs = arch_data[:, 3]
            converged = arch_data[:, 4]
            iterations = arch_data[:, 5]
            improvement_rates = arch_data[:, 6]

            # Compute statistics
            stats[(depth, width)] = {
                'n_runs': len(final_costs),
                'mean_cost': np.mean(final_costs),
                'std_cost': np.std(final_costs, ddof=1),  # Sample std dev
                'min_cost': np.min(final_costs),
                'max_cost': np.max(final_costs),
                'cv_cost': np.std(final_costs, ddof=1) / np.mean(final_costs) if np.mean(final_costs) > 0 else 0,
                'success_rate': np.sum(converged) / len(converged) * 100,
                'mean_improvement_rate': np.mean(improvement_rates),
            }

    return stats

def print_summary_table(stats):
    """Print summary table to console"""
    if stats is None:
        return

    print("\nRobust Statistical Analysis: Spiral Network Architectures\n")

    header = f"{'Architecture':<20} {'Mean Cost':<12} {'Std Dev':<12} {'CV':<8} {'Min':<10} {'Max':<10} {'Success %':<10}"
    print(header)
    print("-" * len(header))

    for (depth, width), st in sorted(stats.items()):
        arch = f"D={depth}, W={width}"
        print(f"{arch:<20} "
              f"{st['mean_cost']:<12.6f} "
              f"{st['std_cost']:<12.6f} "
              f"{st['cv_cost']:<8.4f} "
              f"{st['min_cost']:<10.6f} "
              f"{st['max_cost']:<10.6f} "
              f"{st['success_rate']:<10.1f}")

    print()

def generate_latex_table(stats, output_file=None):
    """
    Generate table 

    Parameters
    ----------
    stats : dict
        Statistics dictionary from compute_statistics()
    output_file : str, optional
        If provided, write to file. Otherwise print to console.
    """
    if stats is None:
        return

    latex_lines = []
    latex_lines.append("% Robust statistical analysis table")
    latex_lines.append("% Generated automatically by analyse_spiral_robust.py")
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\begin{tabular}{lcccccc}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Architecture} & \\textbf{Mean Cost} & \\textbf{Std Dev} & \\textbf{CV} & \\textbf{Best} & \\textbf{Worst} & \\textbf{Success} \\\\")
    latex_lines.append("\\hline")

    for (depth, width), st in sorted(stats.items()):
        # Generate architecture string
        arch_str = f"(2,{''.join([str(width)+',']*depth)}1)"

        # Format with uncertainty
        mean_str = f"{st['mean_cost']:.4f}"
        std_str = f"{st['std_cost']:.4f}"
        cv_str = f"{st['cv_cost']:.2f}"
        min_str = f"{st['min_cost']:.4f}"
        max_str = f"{st['max_cost']:.4f}"
        success_str = f"{st['success_rate']:.0f}\\%"

        # Highlight best architecture (lowest mean cost)
        if st['mean_cost'] < 0.05:  # Threshold for "good" performance
            arch_str = f"\\textbf{{{arch_str}}}"
            mean_str = f"\\textbf{{{mean_str}}}"

        latex_lines.append(f"{arch_str} & ${mean_str} \\pm {std_str}$ & {cv_str} & {min_str} & {max_str} & {success_str} \\\\")

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{Robust statistical analysis of spiral network architectures over 5 independent runs. "
                      "CV = coefficient of variation (std/mean), "
                      "Best/Worst = minimum/maximum cost across repeats, "
                      "Success = percentage of runs converging to $\\tau = 10^{-3}$.}")
    latex_lines.append("\\label{tab:spiral_robust}")
    latex_lines.append("\\end{table}")

    latex_str = "\n".join(latex_lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex_str)
        print(f"\nLaTeX table written to: {output_file}")
    else:
        print("\nLaTeX Table\n")
        print(latex_str)
        print()

def generate_compact_beamer_table(stats, output_file=None):
    """
    Generate compact LaTeX table suitable for Beamer presentation
    """
    if stats is None:
        return

    latex_lines = []
    latex_lines.append("% Compact table for Beamer presentation")
    latex_lines.append("\\begin{table}")
    latex_lines.append("\\footnotesize")
    latex_lines.append("\\begin{tabular}{lccc}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Architecture} & \\textbf{Mean $\\pm$ Std} & \\textbf{Best} & \\textbf{Success} \\\\")
    latex_lines.append("\\hline")

    for (depth, width), st in sorted(stats.items()):
        # Generate architecture string
        layers_str = ','.join([str(width)]*depth)
        arch_str = f"$(2,{layers_str},1)$"

        # Format compactly
        mean_pm_std = f"${st['mean_cost']:.3f} \\pm {st['std_cost']:.3f}$"
        best_str = f"{st['min_cost']:.3f}"
        success_str = f"{st['success_rate']:.0f}\\%"

        # Highlight best
        if st['mean_cost'] < 0.05:
            arch_str = f"\\textbf{{{arch_str.strip('$')}}}$"
            best_str = f"\\textbf{{{best_str}}}"

        latex_lines.append(f"{arch_str} & {mean_pm_std} & {best_str} & {success_str} \\\\")

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    latex_str = "\n".join(latex_lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex_str)
        print(f"Beamer table written to: {output_file}")
    else:
        print("\nBeamer Table (Compact)\n")
        print(latex_str)
        print()

if __name__ == "__main__":
    print("Analysing robust spiral network results\n")

    # Load data
    data = load_robust_results()

    if data is None:
        sys.exit(1)

    # Compute statistics
    stats = compute_statistics(data)

    # Print summary
    print_summary_table(stats)

    # Generate LaTeX tables
    latex_output = config.DATA_OUTPUT / "spiral_robust_table.tex"
    generate_latex_table(stats, latex_output)

    beamer_output = config.DATA_OUTPUT / "spiral_robust_table_beamer.tex"
    generate_compact_beamer_table(stats, beamer_output)

    print("\nAnalysis complete")
