/// Benchmark driver for Exercise 4: Algorithm Optimisation
/// Compares standard training (Algorithm 2) with optimised interlaced training
/// Tests various network architectures on spiral classification problem
/// Measures training time to document speedup

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include "../project2_a.h"

int main()
{
  std::cout << "Exercise 4: Training Algorithm Optimisation Benchmark" << std::endl;
  std::cout << std::endl;

  // Create activation function (shared by all networks)
  ActivationFunction* activation_function_pt = new TanhActivationFunction;

  // Read training data once (same for all networks)
  std::cout << "Reading spiral training data" << std::endl;
  std::vector<std::pair<DoubleVector, DoubleVector>> training_data;

  // Build temporary network to read data
  unsigned n_input = 2;
  std::vector<std::pair<unsigned, ActivationFunction*>> temp_layers;
  temp_layers.push_back(std::make_pair(3, activation_function_pt));
  temp_layers.push_back(std::make_pair(1, activation_function_pt));
  NeuralNetwork temp_network(n_input, temp_layers);

  temp_network.read_training_data(
    "data/input/spiral_training_data.dat",
    training_data);
  std::cout << "Training data points: " << training_data.size() << std::endl;
  std::cout << std::endl;

  // Training parameters (same for all networks)
  double learning_rate = 0.01;
  unsigned n_iterations = 50000;  // Fixed iterations for fair timing comparison

  std::cout << "Training parameters:" << std::endl;
  std::cout << "  learning_rate = " << learning_rate << std::endl;
  std::cout << "  iterations = " << n_iterations << std::endl;
  std::cout << std::endl;

  // Define network architectures to test
  std::vector<std::string> network_names;
  std::vector<std::vector<unsigned>> network_configs;

  // Choose widths and depths to test
  std::vector<unsigned> widths = {16, 32, 64}; // neurons per hidden layer
  std::vector<unsigned> depths = {2, 3};   // number of hidden layers

  for (unsigned d : depths) {
      for (unsigned w : widths) {

          // Build architecture vector: d hidden layers of width w + output 1
          std::vector<unsigned> layers(d, w);
          layers.push_back(1);

          // Build name: depth_X_width_Y
          network_names.push_back(
              "depth_" + std::to_string(d) + "_width_" + std::to_string(w)
          );

          network_configs.push_back(layers);
      }
  }

  // Open results file
  std::ofstream results_file("data/output/timing_comparison.dat");
  results_file << "# Network  StandardTime(s)  OptimisedTime(s)  Speedup" << std::endl;

  unsigned n_repeats = 3;

  // Benchmark each network configuration
  for (unsigned config_id = 0; config_id < network_names.size(); config_id++)
  {
    for (unsigned rep = 0; rep < n_repeats; rep++)
    {
      std::cout << "Network " << (config_id + 1) << " of "
                << network_names.size() << ": " << network_names[config_id] << std::endl;
      std::cout << "Repeat " << (rep + 1) << " of " << n_repeats << std::endl;
      std::cout << "Architecture: (2";
      for (unsigned i = 0; i < network_configs[config_id].size(); i++)
      {
        std::cout << ", " << network_configs[config_id][i];
      }
      std::cout << ")" << std::endl;
      std::cout << std::endl;

      // Build network layers
      std::vector<std::pair<unsigned, ActivationFunction*>> layers;
      for (unsigned i = 0; i < network_configs[config_id].size(); i++)
      {
        layers.push_back(std::make_pair(network_configs[config_id][i],
                                         activation_function_pt));
      }

      // Test 1: Standard training (Algorithm 2)
      std::cout << "Testing standard training" << std::endl;
      NeuralNetwork network_standard(n_input, layers);
      network_standard.initialise_parameters(0.0, 0.1);

      double initial_cost_standard = network_standard.cost_for_training_data(training_data);
      std::cout << "Initial cost: " << initial_cost_standard << std::endl;

      auto start_standard = std::chrono::high_resolution_clock::now();
      network_standard.train(training_data, learning_rate, -1.0, n_iterations, "");
      auto end_standard = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_standard = end_standard - start_standard;

      double final_cost_standard = network_standard.cost_for_training_data(training_data);
      std::cout << "Final cost: " << final_cost_standard << std::endl;
      std::cout << "Time: " << elapsed_standard.count() << " seconds" << std::endl;
      std::cout << std::endl;

      // Test 2: Optimised training (interlaced algorithm)
      std::cout << "Testing optimised training" << std::endl;
      NeuralNetwork network_optimised(n_input, layers);
      network_optimised.initialise_parameters(0.0, 0.1);

      double initial_cost_optimised = network_optimised.cost_for_training_data(training_data);
      std::cout << "Initial cost: " << initial_cost_optimised << std::endl;

      auto start_optimised = std::chrono::high_resolution_clock::now();
      network_optimised.train_optimised(training_data, learning_rate, -1.0, n_iterations, "");
      auto end_optimised = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_optimised = end_optimised - start_optimised;

      double final_cost_optimised = network_optimised.cost_for_training_data(training_data);
      std::cout << "Final cost: " << final_cost_optimised << std::endl;
      std::cout << "Time: " << elapsed_optimised.count() << " seconds" << std::endl;
      std::cout << std::endl;

      // Compute speedup
      double speedup = elapsed_standard.count() / elapsed_optimised.count();

      std::cout << "Speedup: " << speedup << "x" << std::endl;
      std::cout << std::endl;

      // Write to results file
      results_file << network_names[config_id] << " "
                   << elapsed_standard.count() << " "
                   << elapsed_optimised.count() << " "
                   << speedup << std::endl;
    }
  }

  results_file.close();

  std::cout << "Benchmark complete" << std::endl;
  std::cout << "Results saved to data/output/timing_comparison.dat" << std::endl;

  // Clean up
  delete activation_function_pt;
  activation_function_pt = 0;

  return 0;
}
