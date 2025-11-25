/// Test driver for Analysis Exercise 1 - Part 2
/// Train 4 networks on project_training_data.dat to assess reliability
/// Network: (2, 3, 3, 1) with TanhActivationFunction
/// All 4 networks trained in same run with different random initial conditions

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include "../project2_a.h"

int main()
{
  std::cout << "Analysis Exercise 1: Reliability Test - 4 Networks" << std::endl;
  std::cout << std::endl;

  // Create activation function (shared by all networks)
  ActivationFunction* activation_function_pt = new TanhActivationFunction;

  // Read training data once (same for all networks)
  std::cout << "Reading training data" << std::endl;
  std::vector<std::pair<DoubleVector, DoubleVector>> training_data;

  // Build a temporary network just to read the data
  unsigned n_input = 2;
  std::vector<std::pair<unsigned, ActivationFunction*>> non_input_layer;
  non_input_layer.push_back(std::make_pair(3, activation_function_pt));
  non_input_layer.push_back(std::make_pair(3, activation_function_pt));
  non_input_layer.push_back(std::make_pair(1, activation_function_pt));
  NeuralNetwork temp_network(n_input, non_input_layer);

  temp_network.read_training_data(
    "data/input/project_training_data.dat",
    training_data);
  std::cout << "Training data points: " << training_data.size() << std::endl;
  std::cout << std::endl;

  // Training parameters (same for all networks)
  double learning_rate = 0.1;
  double tol = 1.0e-4;
  unsigned max_iter = 100000;

  std::cout << "Training parameters (same for all 4 networks):" << std::endl;
  std::cout << "  learning_rate = " << learning_rate << std::endl;
  std::cout << "  tolerance = " << tol << std::endl;
  std::cout << "  max_iter = " << max_iter << std::endl;
  std::cout << std::endl;

  // Train 4 networks with different random initialisations
  const unsigned n_networks = 4;
  std::vector<double> initial_costs(n_networks);
  std::vector<double> final_costs(n_networks);

  for (unsigned run = 0; run < n_networks; run++)
  {
    std::cout << "========================================" << std::endl;
    std::cout << "Network " << (run + 1) << " of " << n_networks << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Build network: (2, 3, 3, 1)
    std::vector<std::pair<unsigned, ActivationFunction*>> layers;
    layers.push_back(std::make_pair(3, activation_function_pt));
    layers.push_back(std::make_pair(3, activation_function_pt));
    layers.push_back(std::make_pair(1, activation_function_pt));
    NeuralNetwork network(n_input, layers);

    // Initialise parameters with random values
    // Note: Each network gets different random values because we're in same run
    std::cout << "Initialising parameters" << std::endl;
    network.initialise_parameters(0.0, 0.1);
    std::cout << "done" << std::endl;

    // Verify randomness: print first few weights from first layer
    std::cout << "First 3 weights from layer 0: ";
    DoubleMatrix& first_layer_weights = network.layer_pt(0)->weight();
    for (unsigned i = 0; i < 3 && i < first_layer_weights.n(); i++)
    {
      std::cout << first_layer_weights(i, 0) << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    // Initial cost
    initial_costs[run] = network.cost_for_training_data(training_data);
    std::cout << "Initial cost: " << initial_costs[run] << std::endl;
    std::cout << std::endl;

    // Create output filenames for this run
    std::stringstream conv_file, network_file, training_file, output_file;
    conv_file << "data/output/convergence_logs/network_" << (run + 1) << "_convergence.dat";
    network_file << "data/output/trained_networks/network_" << (run + 1) << ".dat";
    training_file << "data/output/network_" << (run + 1) << "_training_data.dat";
    output_file << "data/output/network_outputs/network_" << (run + 1) << "_output.dat";

    // Train
    std::cout << "Training" << std::endl;
    network.train(training_data, learning_rate, tol, max_iter,
                  conv_file.str());
    std::cout << std::endl;

    // Final cost
    final_costs[run] = network.cost_for_training_data(training_data);
    std::cout << "Final cost: " << final_costs[run] << std::endl;
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

    // Summary for this network
    std::cout << "Summary for Network " << (run + 1) << ":" << std::endl;
    std::cout << "  Initial cost: " << initial_costs[run] << std::endl;
    std::cout << "  Final cost: " << final_costs[run] << std::endl;
    std::cout << "  Cost reduction: " << (initial_costs[run] - final_costs[run]) << std::endl;

    if (final_costs[run] < tol)
    {
      std::cout << "  Status: Converged successfully" << std::endl;
    }
    else
    {
      std::cout << "  Status: Did not reach target tolerance" << std::endl;
    }
    std::cout << std::endl;
  }

  // Overall comparison
  std::cout << "========================================" << std::endl;
  std::cout << "COMPARISON OF ALL 4 NETWORKS" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::endl;

  std::cout << "Initial costs:" << std::endl;
  for (unsigned run = 0; run < n_networks; run++)
  {
    std::cout << "  Network " << (run + 1) << ": " << initial_costs[run] << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Final costs:" << std::endl;
  for (unsigned run = 0; run < n_networks; run++)
  {
    std::cout << "  Network " << (run + 1) << ": " << final_costs[run] << std::endl;
  }
  std::cout << std::endl;

  // Calculate statistics
  double min_final = final_costs[0];
  double max_final = final_costs[0];
  double sum_final = 0.0;

  for (unsigned run = 0; run < n_networks; run++)
  {
    sum_final += final_costs[run];
    if (final_costs[run] < min_final)
    {
      min_final = final_costs[run];
    }
    if (final_costs[run] > max_final)
    {
      max_final = final_costs[run];
    }
  }

  double mean_final = sum_final / n_networks;

  // Calculate standard deviation
  double sum_sq_diff = 0.0;
  for (unsigned run = 0; run < n_networks; run++)
  {
    double diff = final_costs[run] - mean_final;
    sum_sq_diff += diff * diff;
  }
  double std_dev = std::sqrt(sum_sq_diff / n_networks);

  std::cout << "Statistics of final costs:" << std::endl;
  std::cout << "  Mean: " << mean_final << std::endl;
  std::cout << "  Std dev: " << std_dev << std::endl;
  std::cout << "  Min: " << min_final << std::endl;
  std::cout << "  Max: " << max_final << std::endl;
  std::cout << "  Range: " << (max_final - min_final) << std::endl;
  std::cout << "  Relative std dev: " << (std_dev / mean_final * 100.0) << "%" << std::endl;
  std::cout << std::endl;

  // Convergence assessment
  unsigned n_converged = 0;
  for (unsigned run = 0; run < n_networks; run++)
  {
    if (final_costs[run] < tol)
    {
      n_converged++;
    }
  }

  std::cout << "Convergence summary:" << std::endl;
  std::cout << "  Networks converged: " << n_converged << " out of " << n_networks << std::endl;
  std::cout << "  Success rate: " << (100.0 * n_converged / n_networks) << "%" << std::endl;
  std::cout << std::endl;

  // Clean up
  delete activation_function_pt;
  activation_function_pt = 0;

  return 0;
}
