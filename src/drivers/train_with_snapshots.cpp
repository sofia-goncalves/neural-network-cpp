/**
 * @file train_with_snapshots.cpp
 * @brief Train (2,16,16,16,1) network and save periodic snapshots for visualization
 *
 * Saves network output at regular intervals to show how decision boundary
 * evolves during training
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "../project2_a.h"

int main()
{
  std::cout << "Training with Snapshots" << std::endl;
  std::cout << "Network: (2,16,16,16,1)" << std::endl;
  std::cout << std::endl;

  // Network architecture: (2, 16, 16, 16, 1)
  unsigned n_input = 2;

  // Activation function
  ActivationFunction* activation_function_pt = new TanhActivationFunction;

  // Build layers: 3 hidden layers of 16 neurons + 1 output neuron
  std::vector<std::pair<unsigned, ActivationFunction*>> layers;
  layers.push_back(std::make_pair(16, activation_function_pt));  // Hidden layer 1
  layers.push_back(std::make_pair(16, activation_function_pt));  // Hidden layer 2
  layers.push_back(std::make_pair(16, activation_function_pt));  // Hidden layer 3
  layers.push_back(std::make_pair(1, activation_function_pt));   // Output layer

  // Create network
  NeuralNetwork network(n_input, layers);
  network.initialise_parameters(0.0, 0.1);

  // Load spiral training data
  std::string training_file = "data/input/spiral_training_data.dat";

  std::vector<std::pair<DoubleVector,DoubleVector>> training_data;
  network.read_training_data(training_file, training_data);

  std::cout << "Loaded " << training_data.size() << " training samples" << std::endl;
  std::cout << std::endl;

  // Training parameters (match successful spiral networks)
  double learning_rate = 0.01;
  double target_tolerance = 1e-3;  // Tighter tolerance for better convergence

  // Snapshot iterations
  std::vector<unsigned> snapshot_iterations = {
    5000, 50000, 500000, 750000, 1000000, 3000000, 6000000
  };

  // Function to save network output on grid
  auto save_snapshot = [&](unsigned iteration) {
    std::string output_file = "data/output/snapshots/snapshot_iter_" +
                              std::to_string(iteration) + ".dat";

    std::ofstream out(output_file);
    if (!out.is_open())
    {
      std::cerr << "Warning: Could not save snapshot to " << output_file << std::endl;
      return;
    }

    // Generate grid of points covering the spiral domain
    double x_min = -1.5, x_max = 1.5;
    double y_min = -1.5, y_max = 1.5;
    unsigned n_points = 200;

    for (unsigned i = 0; i < n_points; i++)
    {
      for (unsigned j = 0; j < n_points; j++)
      {
        double x1 = x_min + (x_max - x_min) * i / (n_points - 1);
        double x2 = y_min + (y_max - y_min) * j / (n_points - 1);

        DoubleVector input(2);
        input[0] = x1;
        input[1] = x2;

        DoubleVector output(1);
        network.feed_forward(input, output);

        out << x1 << " " << x2 << " " << output[0] << std::endl;
      }
    }

    out.close();
    std::cout << "  Saved snapshot at iteration " << iteration << std::endl;
  };

  // Training loop - use the high-level train() method but interrupt for snapshots
  std::cout << "Starting training..." << std::endl;
  std::cout << std::endl;

  std::string conv_file = "data/output/convergence_logs/snapshot_training_convergence.dat";

  // Open convergence file for writing (will be appended to after first chunk)
  std::ofstream conv_stream;
  conv_stream.open(conv_file.c_str());
  if (!conv_stream)
  {
    std::cerr << "Warning: Could not open convergence file: " << conv_file << std::endl;
  }
  conv_stream << "# Iteration  TotalCost" << std::endl;
  conv_stream.close();

  for (unsigned s = 0; s < snapshot_iterations.size(); s++)
  {
    unsigned start_iter = (s == 0) ? 0 : snapshot_iterations[s-1];
    unsigned end_iter = snapshot_iterations[s];
    unsigned iter_chunk = end_iter - start_iter;

    std::cout << "Training from iteration " << start_iter << " to " << end_iter << std::endl;

    // Create temporary convergence file for this chunk
    std::string temp_conv_file = "data/output/convergence_logs/temp_chunk_convergence.dat";

    // Train for this chunk with temporary convergence file
    network.train(training_data, learning_rate, target_tolerance, iter_chunk, temp_conv_file);

    // Append this chunk's convergence data to main file
    std::ifstream temp_in(temp_conv_file.c_str());
    conv_stream.open(conv_file.c_str(), std::ios::app);  // Open in append mode

    if (temp_in && conv_stream)
    {
      std::string line;
      unsigned line_count = 0;
      while (std::getline(temp_in, line))
      {
        // Skip header line
        if (line_count > 0 && !line.empty() && line[0] != '#')
        {
          // Parse iteration and cost
          std::istringstream iss(line);
          unsigned iter_in_chunk;
          double cost;
          if (iss >> iter_in_chunk >> cost)
          {
            // Adjust iteration number to be absolute (not relative to chunk)
            unsigned absolute_iter = start_iter + iter_in_chunk;
            conv_stream << absolute_iter << " " << cost << std::endl;
          }
        }
        line_count++;
      }
    }
    temp_in.close();
    conv_stream.close();

    // Save snapshot
    save_snapshot(end_iter);

    // Check if converged
    double current_cost = network.cost_for_training_data(training_data);
    std::cout << "Iteration " << std::setw(7) << end_iter
              << " | Cost: " << std::scientific << std::setprecision(6) << current_cost;

    if (current_cost < target_tolerance)
    {
      std::cout << " | CONVERGED" << std::endl;
      break;
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << "Training complete" << std::endl;
  std::cout << "Snapshots saved to data/output/snapshots/" << std::endl;

  // Cleanup
  delete activation_function_pt;
  activation_function_pt = 0;

  return 0;
}
