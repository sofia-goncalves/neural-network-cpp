"""
Configuration file for Neural Network Project
Centralised path management for all scripts
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()

# Directory paths
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots"
REPORT_DIR = PROJECT_ROOT / "report"
BUILD_DIR = PROJECT_ROOT / "build"

# Data subdirectories
DATA_INPUT = DATA_DIR / "input"
DATA_OUTPUT = DATA_DIR / "output"
TRAINED_NETWORKS = DATA_OUTPUT / "trained_networks"
CONVERGENCE_LOGS = DATA_OUTPUT / "convergence_logs"
NETWORK_OUTPUTS = DATA_OUTPUT / "network_outputs"

# Plot subdirectories
FIGURES_DIR = PLOTS_DIR / "figures"

# Input data files
SIMPLE_TRAINING_DATA = DATA_INPUT / "project_training_data.dat"
SPIRAL_TRAINING_DATA = DATA_INPUT / "spiral_training_data.dat"
TEST_DATA = DATA_INPUT / "project_test_data.dat"

# Executables
SIMPLE_TEST_EXE = BUILD_DIR / "simple_test"
SPIRAL_TEST_EXE = BUILD_DIR / "spiral_test"
TIMING_TEST_EXE = BUILD_DIR / "timing_test"

# Create output directories if they don't exist
for directory in [DATA_OUTPUT, TRAINED_NETWORKS, CONVERGENCE_LOGS,
                  NETWORK_OUTPUTS, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Print configuration (for debugging)
if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data input: {DATA_INPUT}")
    print(f"Data output: {DATA_OUTPUT}")
    print(f"Figures: {FIGURES_DIR}")
