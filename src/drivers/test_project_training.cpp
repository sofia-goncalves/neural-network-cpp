/// Test driver for Analysis Exercise 1
/// Train network on project_training_data.dat
/// Network: (2, 3, 3, 1) with TanhActivationFunction

#include <iostream>
#include <fstream>
#include "../project2_a.h"

int main()
{
  std::cout << "Analysis Exercise 1: Simple Test Case" << std::endl;
  std::cout << std::endl;

  // Create activation function
  ActivationFunction* activation_function_pt = new TanhActivationFunction;

  // Build network: (2, 3, 3, 1) as specified in project
  unsigned n_input = 2;
  std::vector<std::pair<unsigned, ActivationFunction*>> non_input_layer;

  // First hidden layer: 3 neurons
  non_input_layer.push_back(std::make_pair(3, activation_function_pt));

  // Second hidden layer: 3 neurons
  non_input_layer.push_back(std::make_pair(3, activation_function_pt));

  // Output layer: 1 neuron
  non_input_layer.push_back(std::make_pair(1, activation_function_pt));

  NeuralNetwork network(n_input, non_input_layer);

  std::cout << "Network: (2, 3, 3, 1)" << std::endl;
  std::cout << "Activation: TanhActivationFunction" << std::endl;
  std::cout << std::endl;

  // Initialise parameters
  std::cout << "Initialising parameters" << std::endl;
  network.initialise_parameters(0.0, 0.1);
  std::cout << "done" << std::endl;
  std::cout << std::endl;

  // Read training data
  std::cout << "Reading training data" << std::endl;
  std::vector<std::pair<DoubleVector, DoubleVector>> training_data;
  network.read_training_data(
    "data/input/project_training_data.dat",
    training_data);
  std::cout << "done" << std::endl;
  std::cout << "Training data points: " << training_data.size() << std::endl;
  std::cout << std::endl;

  // Initial cost
  double initial_cost = network.cost_for_training_data(training_data);
  std::cout << "Initial cost: " << initial_cost << std::endl;
  std::cout << std::endl;

  // Training parameters from project
  double learning_rate = 0.1;
  double tol = 1.0e-4;
  unsigned max_iter = 100000;

  std::cout << "Training parameters:" << std::endl;
  std::cout << "  learning_rate = " << learning_rate << std::endl;
  std::cout << "  tolerance = " << tol << std::endl;
  std::cout << "  max_iter = " << max_iter << std::endl;
  std::cout << std::endl;

  std::cout << "Training" << std::endl;
  network.train(training_data, learning_rate, tol, max_iter,
                "data/output/convergence_logs/simple_test_convergence.dat");
  std::cout << std::endl;

  // Final cost
  double final_cost = network.cost_for_training_data(training_data);
  std::cout << "Final cost: " << final_cost << std::endl;
  std::cout << std::endl;

  // Save trained network
  std::cout << "Saving trained network parameters" << std::endl;
  network.write_parameters_to_disk("data/output/trained_networks/simple_test_network.dat");
  std::cout << "done" << std::endl;
  std::cout << std::endl;

  // Output training data for plotting
  std::cout << "Saving training data for plotting" << std::endl;
  std::ofstream training_data_file("data/output/simple_test_training_data.dat");
  network.output_training_data(training_data_file, training_data);
  training_data_file.close();
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

  network.output("data/output/network_outputs/simple_test_output.dat", grid_points);
  std::cout << "done" << std::endl;
  std::cout << std::endl;

  // Summary
  std::cout << "Summary:" << std::endl;
  std::cout << "  Initial cost: " << initial_cost << std::endl;
  std::cout << "  Final cost: " << final_cost << std::endl;
  std::cout << "  Cost reduction: " << (initial_cost - final_cost) << std::endl;
  std::cout << std::endl;

  if (final_cost < tol)
  {
    std::cout << "Training converged successfully" << std::endl;
  }
  else
  {
    std::cout << "Training did not reach target tolerance" << std::endl;
    std::cout << "Consider running for more iterations or adjusting learning rate" << std::endl;
  }

  // Clean up
  delete activation_function_pt;
  activation_function_pt = 0;

  return 0;
}
