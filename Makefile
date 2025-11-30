# Compiler settings
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -Wno-unused-variable
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

# Executables (derived from driver files)
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

# Clean build artifacts
clean:
	rm -f $(BUILD_DIR)/*
	rm -f $(SRC_DIR)/*.o

# Clean all generated data
clean_data:
	rm -rf $(DATA_DIR)/output/trained_networks/*
	rm -rf $(DATA_DIR)/output/convergence_logs/*
	rm -rf $(DATA_DIR)/output/network_outputs/*
	rm -rf $(DATA_DIR)/output/snapshots/*
	rm -f $(DATA_DIR)/output/*.dat
	rm -f $(DATA_DIR)/output/*.tex

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
	@echo "  clean            Remove build artifacts"
	@echo "  clean_data       Remove generated data"
	@echo "  clean_plots      Remove generated plots"
	@echo "  distclean        Full clean"
	@echo "  help             Show this message"

.PHONY: all debug clean clean_data clean_plots distclean help
