/// Test driver for Analysis Exercise 2 - Spiral Training Data
/// Tests various network architectures on the more challenging spiral classification problem
/// Varies depths and widths in turn

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include "../project2_a.h"

int main()
{
  std::cout << "Analysis Exercise 2: Spiral Classification Problem" << std::endl;
  std::cout << std::endl;

  // Create activation function (shared by all networks)
  ActivationFunction* activation_function_pt = new TanhActivationFunction;

  // Read training data once (same for all networks)
  std::cout << "Reading spiral training data" << std::endl;
  std::vector<std::pair<DoubleVector, DoubleVector>> training_data;

  // Build a temporary network just to read the data
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
  double tol = 1.0e-3;
  unsigned max_iter = 4000000;

  std::cout << "Training parameters:" << std::endl;
  std::cout << "  learning_rate = " << learning_rate << std::endl;
  std::cout << "  tolerance = " << tol << std::endl;
  std::cout << "  max_iter = " << max_iter << std::endl;
  std::cout << std::endl;

  // Define network architectures to test
  // Specify which depths (number of hidden layers) and widths (neurons per layer) to test
  std::vector<unsigned> depths_to_test = {2, 3, 4};
  std::vector<unsigned> widths_to_test = {4, 8, 16};

  // Automatically generate all combinations
  std::vector<std::string> network_names;
  std::vector<std::vector<unsigned>> network_configs;

  for (unsigned depth : depths_to_test)
  {
    for (unsigned width : widths_to_test)
    {
      // Create name: e.g., "depth_3_width_4"
      std::stringstream name_ss;
      name_ss << "depth_" << depth << "_width_" << width;
      network_names.push_back(name_ss.str());

      // Create architecture: (width, width, ..., 1) with 'depth' hidden layers
      std::vector<unsigned> config;
      for (unsigned i = 0; i < depth; i++)
      {
        config.push_back(width);
      }
      config.push_back(1);  // Output layer
      network_configs.push_back(config);
    }
  }

  // Train each network configuration
  for (unsigned config_id = 0; config_id < network_names.size(); config_id++)
  {
    std::cout << "========================================" << std::endl;
    std::cout << "Network Configuration " << (config_id + 1) << " of "
              << network_names.size() << std::endl;
    std::cout << "Name: " << network_names[config_id] << std::endl;
    std::cout << "Architecture: (2";
    for (unsigned i = 0; i < network_configs[config_id].size(); i++)
    {
      std::cout << ", " << network_configs[config_id][i];
    }
    std::cout << ")" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Build network with current configuration
    std::vector<std::pair<unsigned, ActivationFunction*>> layers;
    for (unsigned i = 0; i < network_configs[config_id].size(); i++)
    {
      layers.push_back(std::make_pair(network_configs[config_id][i],
                                       activation_function_pt));
    }
    NeuralNetwork network(n_input, layers);

    // Initialise parameters with random values
    std::cout << "Initialising parameters" << std::endl;
    network.initialise_parameters(0.0, 0.1);
    std::cout << "done" << std::endl;
    std::cout << std::endl;

    // Initial cost
    double initial_cost = network.cost_for_training_data(training_data);
    std::cout << "Initial cost: " << initial_cost << std::endl;
    std::cout << std::endl;

    // Create output filenames
    std::stringstream conv_file, network_file, training_file, output_file;
    conv_file << "data/output/convergence_logs/spiral_"
              << network_names[config_id] << "_convergence.dat";
    network_file << "data/output/trained_networks/spiral_"
                 << network_names[config_id] << ".dat";
    training_file << "data/output/spiral_"
                  << network_names[config_id] << "_training_data.dat";
    output_file << "data/output/network_outputs/spiral_"
                << network_names[config_id] << "_output.dat";

    // Train
    std::cout << "Training (this may take several minutes)" << std::endl;
    network.train(training_data, learning_rate, tol, max_iter,
                  conv_file.str());
    std::cout << std::endl;

    // Final cost
    double final_cost = network.cost_for_training_data(training_data);
    std::cout << "Final cost: " << final_cost << std::endl;
    std::cout << std::endl;

    // Save trained network
    std::cout << "Saving trained network parameters" << std::endl;
    network.write_parameters_to_disk(network_file.str());
    std::cout << "done" << std::endl;
    std::cout << std::endl;

    // Output training data for plotting
    std::cout << "Saving training data for plotting" << std::endl;
    std::ofstream train_file(training_file.str().c_str());
    network.output_training_data(train_file, training_data);
    train_file.close();
    std::cout << "done" << std::endl;
    std::cout << std::endl;

    // Output network predictions on grid for plotting
    std::cout << "Generating output for plotting" << std::endl;

    // Create grid of points
    std::vector<DoubleVector> grid_points;
    unsigned n_grid = 100;
    for (unsigned i = 0; i < n_grid; i++)
    {
      for (unsigned j = 0; j < n_grid; j++)
      {
        DoubleVector point(2);
        point[0] = static_cast<double>(i) / (n_grid - 1);
        point[1] = static_cast<double>(j) / (n_grid - 1);
        grid_points.push_back(point);
      }
    }

    network.output(output_file.str(), grid_points);
    std::cout << "done" << std::endl;
    std::cout << std::endl;

    // Summary
    std::cout << "Summary for " << network_names[config_id] << ":" << std::endl;
    std::cout << "  Initial cost: " << initial_cost << std::endl;
    std::cout << "  Final cost: " << final_cost << std::endl;
    std::cout << "  Cost reduction: " << (initial_cost - final_cost) << std::endl;

    if (final_cost < tol)
    {
      std::cout << "  Status: Converged successfully" << std::endl;
    }
    else
    {
      std::cout << "  Status: Did not reach target tolerance" << std::endl;
      std::cout << "  (Final cost " << final_cost << " > tolerance " << tol << ")" << std::endl;
    }
    std::cout << std::endl;
  }


  // Clean up
  delete activation_function_pt;
  activation_function_pt = 0;

  return 0;
}
