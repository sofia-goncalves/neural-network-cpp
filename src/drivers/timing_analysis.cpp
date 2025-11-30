/// Timing Analysis for Analysis Exercise 3 - Computational Cost Assessment
/// Measures actual computation time for:
///   1. Feed-forward operation
///   2. Gradient computation via finite-differencing
///   3. Gradient computation via back-propagation
///
/// Tests networks with varying N_layer and N_neuron to validate theoretical
/// complexity predictions: Cost ~ N^a_layer × N^b_neuron
///
/// Output: data/output/timing_results.dat with timing data for all configurations

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include "../project2_a.h"

// Timing utilities
using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Duration = std::chrono::duration<double>;

/// Measure time for N_repeat feed-forward operations
double time_feedforward(NeuralNetwork& network,
                        const DoubleVector& input,
                        const unsigned& n_repeat)
{
  DoubleVector output;

  TimePoint start = Clock::now();
  for (unsigned i = 0; i < n_repeat; i++)
  {
    network.feed_forward(input, output);
  }
  TimePoint end = Clock::now();

  Duration elapsed = end - start;
  return elapsed.count() / n_repeat;  // Average time per operation
}

/// Measure time for N_repeat finite-difference gradient computations
double time_finite_difference(NeuralNetwork& network,
                               const DoubleVector& input,
                               const DoubleVector& target,
                               const unsigned& n_repeat)
{
  std::vector<DoubleMatrix> grad_W;
  std::vector<DoubleVector> grad_b;

  TimePoint start = Clock::now();
  for (unsigned i = 0; i < n_repeat; i++)
  {
    network.test_compute_gradients_finite_difference(input, target, grad_W, grad_b);
  }
  TimePoint end = Clock::now();

  Duration elapsed = end - start;
  return elapsed.count() / n_repeat;  // Average time per operation
}

/// Measure time for N_repeat back-propagation gradient computations
double time_backprop(NeuralNetwork& network,
                     const DoubleVector& input,
                     const DoubleVector& target,
                     const unsigned& n_repeat)
{
  std::vector<DoubleMatrix> grad_W;
  std::vector<DoubleVector> grad_b;

  TimePoint start = Clock::now();
  for (unsigned i = 0; i < n_repeat; i++)
  {
    network.test_compute_gradients_backprop(input, target, grad_W, grad_b);
  }
  TimePoint end = Clock::now();

  Duration elapsed = end - start;
  return elapsed.count() / n_repeat;  // Average time per operation
}

int main()
{
  std::cout << "Analysis Exercise 3: Computational Cost Assessment" << std::endl;
  std::cout << "Timing analysis to validate theoretical complexity predictions" << std::endl;
  std::cout << std::endl;

  // Create activation function (shared by all networks)
  ActivationFunction* activation_function_pt = new TanhActivationFunction;

  // Test configurations
  // Vary N_layer: number of hidden layers (total layers = N_layer + 1 including output)
  std::vector<unsigned> n_layer_values = {2, 3, 4, 5, 6};

  // Vary N_neuron: neurons per layer
  std::vector<unsigned> n_neuron_values = {16, 32, 64, 128, 256};

  // Number of repeats for each timing measurement (for statistical stability)
  const unsigned N_TIMING_REPEATS = 1000;

  // Test data dimensions
  const unsigned N_INPUT = 2;  // 2D input (like spiral problem)
  const unsigned N_OUTPUT = 1; // Binary classification

  // Create dummy input and target for timing
  DoubleVector test_input(N_INPUT);
  test_input[0] = 0.5;
  test_input[1] = -0.3;

  DoubleVector test_target(N_OUTPUT);
  test_target[0] = 1.0;

  // Open output file
  std::ofstream timing_file("data/output/timing_results.dat");
  timing_file << "# Timing analysis results" << std::endl;
  timing_file << "# Columns: n_layer n_neuron n_params time_feedforward time_fd time_backprop ratio_fd_to_ff ratio_bp_to_ff ratio_fd_to_bp" << std::endl;
  timing_file << std::setprecision(10) << std::scientific;

  std::cout << "Configuration:" << std::endl;
  std::cout << "  N_input = " << N_INPUT << std::endl;
  std::cout << "  N_output = " << N_OUTPUT << std::endl;
  std::cout << "  N_timing_repeats = " << N_TIMING_REPEATS << std::endl;
  std::cout << std::endl;

  std::cout << "Testing " << n_layer_values.size() << " × "
            << n_neuron_values.size() << " = "
            << (n_layer_values.size() * n_neuron_values.size())
            << " configurations" << std::endl;
  std::cout << std::endl;

  unsigned config_count = 0;
  unsigned total_configs = n_layer_values.size() * n_neuron_values.size();

  // Test all combinations of N_layer and N_neuron
  for (unsigned n_layer : n_layer_values)
  {
    for (unsigned n_neuron : n_neuron_values)
    {
      config_count++;

      std::cout << "Configuration " << config_count << "/" << total_configs
                << ": N_layer=" << n_layer << ", N_neuron=" << n_neuron << std::flush;

      // Build network architecture
      // Structure: (N_INPUT, N_neuron, N_neuron, ..., N_neuron, N_OUTPUT)
      //            with n_layer hidden layers of size N_neuron
      std::vector<std::pair<unsigned, ActivationFunction*>> layers;

      // Add n_layer hidden layers
      for (unsigned i = 0; i < n_layer; i++)
      {
        layers.push_back(std::make_pair(n_neuron, activation_function_pt));
      }

      // Add output layer
      layers.push_back(std::make_pair(N_OUTPUT, activation_function_pt));

      // Create network
      NeuralNetwork network(N_INPUT, layers);

      // Initialize with random parameters
      network.initialise_parameters(0.0, 0.1);

      // Count total parameters for reference
      unsigned n_params = 0;
      // First hidden layer: (N_INPUT × N_neuron) weights + N_neuron biases
      n_params += N_INPUT * n_neuron + n_neuron;
      // Subsequent hidden layers: (N_neuron × N_neuron) weights + N_neuron biases
      for (unsigned i = 1; i < n_layer; i++)
      {
        n_params += n_neuron * n_neuron + n_neuron;
      }
      // Output layer: (N_neuron × N_OUTPUT) weights + N_OUTPUT biases
      n_params += n_neuron * N_OUTPUT + N_OUTPUT;

      // === TIMING MEASUREMENTS ===

      // 1. Feed-forward timing
      double time_ff = time_feedforward(network, test_input, N_TIMING_REPEATS);

      // 2. Finite-difference gradient timing
      // Note: For large networks, FD can be very slow, so reduce repeats if needed
      unsigned n_repeat_fd = N_TIMING_REPEATS;
      if (n_params > 1000) n_repeat_fd = 10;  // Reduce for large networks
      if (n_params > 5000) n_repeat_fd = 1;   // Single run for very large networks

      double time_fd = time_finite_difference(network, test_input, test_target, n_repeat_fd);

      // 3. Back-propagation gradient timing
      double time_bp = time_backprop(network, test_input, test_target, N_TIMING_REPEATS);

      // Compute ratios for comparison
      double ratio_fd_to_ff = time_fd / time_ff;
      double ratio_bp_to_ff = time_bp / time_ff;
      double ratio_fd_to_bp = time_fd / time_bp;

      // Write to file
      timing_file << n_layer << " "
                  << n_neuron << " "
                  << n_params << " "
                  << time_ff << " "
                  << time_fd << " "
                  << time_bp << " "
                  << ratio_fd_to_ff << " "
                  << ratio_bp_to_ff << " "
                  << ratio_fd_to_bp << std::endl;

      std::cout << " done (FF: " << time_ff << "s, FD: " << time_fd
                << "s, BP: " << time_bp << "s)" << std::endl;
    }
  }

  timing_file.close();

  std::cout << std::endl;
  std::cout << "Timing analysis complete" << std::endl;

  // Cleanup
  delete activation_function_pt;
  activation_function_pt = 0;

  return 0;
}