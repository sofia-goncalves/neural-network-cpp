#!/usr/bin/env python3
"""
Shared plotting utilities for neural network visualisation
"""

import numpy as np
import matplotlib.pyplot as plt

def read_training_data(filepath):
    """Read training data in format: x1 x2 target"""
    data = np.loadtxt(filepath)

    # Split into inputs and targets
    inputs = data[:, :2]
    targets = data[:, 2]

    return inputs, targets

def read_network_output(filepath):
    """Read network output in format: x1 x2 output"""
    data = np.loadtxt(filepath)
    return data

def prepare_contour_data(network_data):
    """
    Prepare data for contour plotting

    C++ generates data as: for each x1, loop over all x2
    So data order is: (x1=0,x2=0..1), (x1=0.01,x2=0..1), ...

    Parameters
    ----------
    network_data : ndarray
        Network output data (N x 3: x1, x2, output)

    Returns
    -------
    X1, X2, Z : ndarrays
        Meshgrid arrays ready for contour plotting
    """
    x1 = network_data[:, 0]
    x2 = network_data[:, 1]
    z = network_data[:, 2]

    # Create grid for contour plot
    x1_unique = np.unique(x1)
    x2_unique = np.unique(x2)
    X1, X2 = np.meshgrid(x1_unique, x2_unique)

    # Reshape to (n_x1, n_x2) then transpose to match meshgrid's (n_x2, n_x1) layout
    Z = z.reshape(len(x1_unique), len(x2_unique)).T

    return X1, X2, Z

def plot_training_data(ax, inputs, targets):
    """
    Plot training data points on given axes

    Parameters
    ----------
    ax : matplotlib axes
        Axes to plot on
    inputs : ndarray
        Input data (N x 2)
    targets : ndarray
        Target labels (N,)
    """
    mask_pos = targets > 0
    mask_neg = targets < 0

    ax.scatter(inputs[mask_pos, 0], inputs[mask_pos, 1],
               c='red', marker='o', s=100, label='Class +1',
               edgecolors='black', linewidths=1.5, zorder=3)
    ax.scatter(inputs[mask_neg, 0], inputs[mask_neg, 1],
               c='blue', marker='x', s=100, label='Class -1',
               linewidths=2, zorder=3)

def plot_decision_boundary(ax, X1, X2, Z, **kwargs):
    """
    Plot decision boundary (zero contour) on given axes

    Parameters
    ----------
    ax : matplotlib axes
        Axes to plot on
    X1, X2, Z : ndarrays
        Meshgrid data from prepare_contour_data
    **kwargs : dict
        Additional arguments passed to contour()
    """
    default_kwargs = {'levels': [0], 'colors': 'black', 'linewidths': 2, 'zorder': 2}
    default_kwargs.update(kwargs)

    return ax.contour(X1, X2, Z, **default_kwargs)

def format_axes(ax, title='Neural Network Decision Boundary', xlim=(0, 1), ylim=(0, 1)):
    """
    Apply standard formatting to axes

    Parameters
    ----------
    ax : matplotlib axes
        Axes to format
    title : str
        Plot title
    xlim, ylim : tuple
        Axis limits
    """
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, zorder=1)
    ax.set_aspect('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
