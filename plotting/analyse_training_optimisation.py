#!/usr/bin/env python3
"""
Analysis script for Exercise 4: Training Algorithm Optimisation

Analyses timing comparison between standard and optimised training algorithms.
Generates plots and tables for the report.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Set up plotting style
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['figure.dpi'] = 450
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13


def expand_arch(name):
    parts = name.split('_')
    depth = int(parts[1])
    width = int(parts[3])
    layers = [2] + [width]*depth + [1]
    return "(" + ", ".join(str(x) for x in layers) + ")"

def load_timing_data(filename):
    """Load timing comparison data from file (with repeats)"""
    data = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) == 4:
                network = parts[0]
                standard_time = float(parts[1])
                optimised_time = float(parts[2])
                speedup = float(parts[3])

                if network not in data:
                    data[network] = {
                        'standard': [],
                        'optimised': [],
                        'speedup': []
                    }

                data[network]['standard'].append(standard_time)
                data[network]['optimised'].append(optimised_time)
                data[network]['speedup'].append(speedup)

    # Convert lists → mean & std
    processed = {}
    for net, vals in data.items():
        processed[net] = {
            'standard': np.mean(vals['standard']),
            'optimised': np.mean(vals['optimised']),
            'speedup': np.mean(vals['speedup']),
            'standard_err': np.std(vals['standard']),
            'optimised_err': np.std(vals['optimised']),
            'speedup_err': np.std(vals['speedup']),
        }

    return processed


def create_timing_comparison_plot(data, output_file):
    """Create bar chart comparing training times with error bars"""
    
    networks = list(data.keys())

    standard_times = [data[n]['standard'] for n in networks]
    optimised_times = [data[n]['optimised'] for n in networks]
    standard_err = [data[n]['standard_err'] for n in networks]
    optimised_err = [data[n]['optimised_err'] for n in networks]

    x = np.arange(len(networks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, standard_times, width, yerr=standard_err,
           label='Standard', color="#4eb2ff", alpha=0.4, capsize=8)
    ax.bar(x + width/2, optimised_times, width, yerr=optimised_err,
           label='Optimised', color="#68ff74a5", alpha=0.4, capsize=8)

    ax.set_xlabel('Network Architecture')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison: Standard vs Optimised Algorithm')
    ax.set_xticks(x)
    ax.set_xticklabels([expand_arch(n) for n in networks],
                    rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved timing comparison plot to {output_file}")
    plt.close()


def create_speedup_plot(data, output_file):
    """Create speedup plot with error bars"""
    networks = list(data.keys())
    speedups = [data[n]['speedup'] for n in networks]
    speedup_err = [data[n]['speedup_err'] for n in networks]

    # Extract network size information
    network_params = []
    for n in networks:
        parts = n.split('_')
        depth = int(parts[1])
        width = int(parts[3])
        params = (2 * width + width)
        params += (depth - 1) * (width * width + width)
        params += (width * 1 + 1)
        network_params.append(params)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(network_params, speedups, yerr=speedup_err, ecolor='gray',
                fmt='bx', lw=0.6, markersize=8, alpha=0.8, capsize=4)

    ax.set_xscale('log')
    ax.set_xlabel('Number of Parameters (log scale)')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('Speedup vs Network Size')
    ax.grid(True, alpha=0.3)
        
    for params, speedup, name in zip(network_params, speedups, networks):
        label = expand_arch(name)
        ax.annotate(label, (params, speedup), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved speedup plot to {output_file}")
    plt.close()


def create_summary_table(data):
    """Create LaTeX summary table with mean values (unchanged output format)"""
    print("\nTiming Results Summary:")
    print("-" * 80)
    print(f"{'Network':<20} {'Standard (s)':<15} {'Optimised (s)':<15} {'Speedup':<10}")
    print("-" * 80)

    for network, values in data.items():
        network_label = network.replace('_', ' ').title()
        print(f"{network_label:<20} {values['standard']:<15.6f} "
              f"{values['optimised']:<15.6f} {values['speedup']:<10.3f}x")

    print("-" * 80)

    avg_speedup = np.mean([v['speedup'] for v in data.values()])
    print(f"\nAverage speedup: {avg_speedup:.3f}x")

    total_standard = sum([v['standard'] for v in data.values()])
    total_optimised = sum([v['optimised'] for v in data.values()])
    total_speedup = total_standard / total_optimised
    print(f"Overall speedup: {total_speedup:.3f}x")

    return avg_speedup


def create_latex_table(data, output_file):
    """Generate LaTeX table using *mean* values (unchanged layout)"""
    with open(output_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Network Architecture & Standard (s) & Optimised (s) & Speedup \\\\\n")
        f.write("\\midrule\n")

        for network, values in data.items():
            parts = network.split('_')
            depth = parts[1]
            width = parts[3]
            arch_label = f"Depth {depth}, Width {width}"

            f.write(f"{arch_label} & "
                   f"{values['standard']:.4f} & "
                   f"{values['optimised']:.4f} & "
                   f"{values['speedup']:.2f}$\\times$ \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Training time comparison between standard and optimised algorithms}\n")
        f.write("\\label{tab:timing_comparison}\n")
        f.write("\\end{table}\n")

    print(f"Saved LaTeX table to {output_file}")


def create_joint_figure(data, output_file):
    """Create a combined figure using EXACT same formatting as the individual plots."""
    
    networks = list(data.keys())

    standard_times = [data[n]['standard'] for n in networks]
    optimised_times = [data[n]['optimised'] for n in networks]
    standard_err = [data[n]['standard_err'] for n in networks]
    optimised_err = [data[n]['optimised_err'] for n in networks]

    x = np.arange(len(networks))
    width = 0.35

    speedups = [data[n]['speedup'] for n in networks]
    speedup_err = [data[n]['speedup_err'] for n in networks]

    network_params = []
    for n in networks:
        parts = n.split('_')
        depth = int(parts[1])
        width_ = int(parts[3])
        params = (2 * width_ + width_)
        params += (depth - 1) * (width_ * width_ + width_)
        params += (width_ * 1 + 1)
        network_params.append(params)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))


    # ax1.bar(x - width/2, standard_times, width, yerr=standard_err,
    #         label='Standard', color="#4eb2ff", alpha=0.4, capsize=8)

    # ax1.bar(x + width/2, optimised_times, width, yerr=optimised_err,
    #         label='Optimised', color="#68ff74a5", alpha=0.4, capsize=8)

    error_cfg = dict(ecolor='black', elinewidth=1, alpha=0.5, capsize=8)

    ax1.bar(
        x - width/2,
        standard_times,
        width,
        yerr=standard_err,
        label='Standard',
        color="#4eb2ff",
        alpha=0.4,
        error_kw=error_cfg
    )

    ax1.bar(
        x + width/2,
        optimised_times,
        width,
        yerr=optimised_err,
        label='Optimised',
        color="#68ff74a5",
        alpha=0.4,
        error_kw=error_cfg
    )

    ax1.set_xlabel('Network Architecture')
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([expand_arch(n) for n in networks],
                        rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ax2.errorbar(network_params, speedups, yerr=speedup_err, ecolor='gray',
                 fmt='bx', lw=0.6, markersize=8, alpha=0.8, capsize=4)

    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Parameters (log scale)')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Speedup vs Network Size')
    ax2.grid(True, alpha=0.3)

    # Identical label annotation
    for params, speedup, name in zip(network_params, speedups, networks):
        label = expand_arch(name)
        ax2.annotate(label, (params, speedup), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved joint figure to {output_file}")



def main():
    print("Exercise 4: Training Algorithm Optimisation Analysis")
    print()

    data_file = 'data/output/timing_comparison.dat'
    plots_dir = 'plots/figures'

    os.makedirs(plots_dir, exist_ok=True)

    print(f"Loading data from {data_file}")
    data = load_timing_data(data_file)
    print(f"Loaded {len(data)} network configurations")
    print()

    avg_speedup = create_summary_table(data)
    print()

    print("Generating plots")
    create_timing_comparison_plot(data,
        os.path.join(plots_dir, 'ex4_timing_comparison.png'))
    create_speedup_plot(data,
        os.path.join(plots_dir, 'ex4_speedup_vs_size.png'))
    create_joint_figure(data,
        os.path.join(plots_dir, 'ex4_joint_figure.png'))
    print()

    latex_file = 'data/output/ex4_timing_table.tex'
    create_latex_table(data, latex_file)
    print()
    

    print("Analysis complete")


if __name__ == '__main__':
    main()
