# Makefile for Neural Network Project (MATH49111 Project 2a)
# Author: Sofia Goncalves
# Student ID: 11058869

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra
OPTFLAGS = -O3
DEBUGFLAGS = -g -DRANGE_CHECKING
LDFLAGS =

# Directories
SRC_DIR = src
DRIVER_DIR = $(SRC_DIR)/drivers
BUILD_DIR = build
DATA_DIR = data

# Source files
HEADERS = $(SRC_DIR)/project2_a.h $(SRC_DIR)/project2_a_basics.h $(SRC_DIR)/dense_linear_algebra.h
DRIVERS = $(wildcard $(DRIVER_DIR)/*.cpp)

# Executables (automatically derived from driver files)
EXECUTABLES = $(patsubst $(DRIVER_DIR)/%.cpp,$(BUILD_DIR)/%,$(DRIVERS))

# Default target: build all executables with optimisation
all: CXXFLAGS += $(OPTFLAGS)
all: $(EXECUTABLES)

# Debug target: build with debugging symbols and range checking
debug: CXXFLAGS += $(DEBUGFLAGS)
debug: clean $(EXECUTABLES)

# Pattern rule: build executable from driver
$(BUILD_DIR)/%: $(DRIVER_DIR)/%.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Individual targets for convenience
simple_test: $(BUILD_DIR)/simple_test
spiral_analysis: $(BUILD_DIR)/spiral_analysis
timing_analysis: $(BUILD_DIR)/timing_analysis
optimisation_test: $(BUILD_DIR)/optimisation_test

# Run targets (examples)
run_simple: $(BUILD_DIR)/simple_test
	./$(BUILD_DIR)/simple_test

run_spiral: $(BUILD_DIR)/spiral_analysis
	./$(BUILD_DIR)/spiral_analysis

# Clean build artifacts
clean:
	rm -f $(BUILD_DIR)/*
	rm -f $(SRC_DIR)/*.o

# Clean all generated data (careful!)
clean_data:
	rm -rf $(DATA_DIR)/output/trained_networks/*
	rm -rf $(DATA_DIR)/output/convergence_logs/*
	rm -rf $(DATA_DIR)/output/network_outputs/*

# Clean generated plots
clean_plots:
	rm -f plots/figures/*.png
	rm -f plots/figures/*.pdf
	rm -f plots/figures/*.eps

# Full clean
distclean: clean clean_data clean_plots

# Help message
help:
	@echo "Neural Network Project Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all              Build all executables (optimised)"
	@echo "  debug            Build with debugging symbols"
	@echo "  simple_test      Build simple test driver"
	@echo "  spiral_analysis  Build spiral analysis driver"
	@echo "  timing_analysis  Build timing analysis driver"
	@echo "  run_simple       Build and run simple test"
	@echo "  run_spiral       Build and run spiral analysis"
	@echo "  clean            Remove build artifacts"
	@echo "  clean_data       Remove generated data"
	@echo "  clean_plots      Remove generated plots"
	@echo "  distclean        Full clean"
	@echo "  help             Show this message"

.PHONY: all debug clean clean_data clean_plots distclean help run_simple run_spiral
