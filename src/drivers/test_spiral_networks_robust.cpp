/// Robust statistical analysis for Analysis Exercise 2 - Spiral Training Data
/// Runs each network architecture N_REPEATS times with different random initializations
/// to obtain mean and standard deviation of final cost (accounts for random initialization effects)
///
/// Output: data/output/spiral_robust_results.dat with statistics for each architecture

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "../project2_a.h"

int main()
{
  std::cout << "Analysis Exercise 2: Robust Statistical Analysis" << std::endl;
  std::cout << "Running multiple repeats to quantify impact of random initialization" << std::endl;
  std::cout << std::endl;

  // Number of repeats for statistical robustness
  const unsigned N_REPEATS = 5;

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
  unsigned max_iter = 10000000;

  std::cout << "Training parameters:" << std::endl;
  std::cout << "  learning_rate = " << learning_rate << std::endl;
  std::cout << "  tolerance = " << tol << std::endl;
  std::cout << "  max_iter = " << max_iter << std::endl;
  std::cout << "  N_REPEATS = " << N_REPEATS << std::endl;
  std::cout << std::endl;

  // Define network architectures to test
  std::vector<unsigned> depths_to_test = {2, 3, 4};
  std::vector<unsigned> widths_to_test = {4, 8, 16};

  // Open output file for robust results
  std::ofstream robust_file("data/output/spiral_robust_results.dat");
  robust_file << "# Robust statistical analysis: " << N_REPEATS << " repeats per architecture" << std::endl;
  robust_file << "# Columns: depth width repeat final_cost converged iterations improvement_rate" << std::endl;
  robust_file << std::setprecision(10);

  // Automatically generate all combinations and run N_REPEATS times
  unsigned total_runs = depths_to_test.size() * widths_to_test.size() * N_REPEATS;
  unsigned current_run = 0;

  for (unsigned depth : depths_to_test)
  {
    for (unsigned width : widths_to_test)
    {
      // Architecture name
      std::stringstream arch_name;
      arch_name << "depth_" << depth << "_width_" << width;

      std::cout << "Architecture: " << arch_name.str() << std::endl;
      std::cout << "Layers: (2";
      for (unsigned i = 0; i < depth; i++) std::cout << ", " << width;
      std::cout << ", 1)" << std::endl;
      std::cout << std::endl;

      // Run N_REPEATS times with different random initializations
      for (unsigned repeat = 0; repeat < N_REPEATS; repeat++)
      {
        current_run++;
        std::cout << "  Repeat " << (repeat + 1) << "/" << N_REPEATS
                  << " (overall: " << current_run << "/" << total_runs << ")" << std::endl;

        // Build network with current configuration
        std::vector<std::pair<unsigned, ActivationFunction*>> layers;
        for (unsigned i = 0; i < depth; i++)
        {
          layers.push_back(std::make_pair(width, activation_function_pt));
        }
        layers.push_back(std::make_pair(1, activation_function_pt));  // Output layer

        NeuralNetwork network(n_input, layers);

        // Initialize with random weights and biases
        network.initialise_parameters(0.0, 0.1);

        // Set up convergence history file (only save for repeat 0 for plotting)
        std::stringstream conv_file;
        if (repeat == 0)
        {
          conv_file << "data/output/convergence_logs/spiral_" << arch_name.str()
                    << "_convergence.dat";
        }

        // Train network
        network.train(training_data, learning_rate, tol, max_iter, conv_file.str());

        // Get final cost
        double final_cost = network.cost_for_training_data(training_data);
        bool converged = (final_cost < tol);

        // Compute improvement rate (cost reduction per 1000 iterations at end)
        // Read last few points from convergence to estimate if still improving
        double improvement_rate = 0.0;
        if (repeat == 0 && !conv_file.str().empty())
        {
          // Read convergence file to get improvement rate
          std::ifstream conv_in(conv_file.str().c_str());
          std::string line;
          std::vector<double> recent_costs;
          std::vector<unsigned> recent_iters;

          // Skip header
          std::getline(conv_in, line);

          // Read all data
          while (std::getline(conv_in, line))
          {
            std::istringstream iss(line);
            unsigned iter;
            double cost;
            if (iss >> iter >> cost)
            {
              recent_iters.push_back(iter);
              recent_costs.push_back(cost);
            }
          }
          conv_in.close();

          // Compute improvement rate from last 10 points
          if (recent_costs.size() >= 10)
          {
            unsigned n = recent_costs.size();
            double cost_early = recent_costs[n - 10];
            double cost_late = recent_costs[n - 1];
            unsigned iter_early = recent_iters[n - 10];
            unsigned iter_late = recent_iters[n - 1];

            if (iter_late > iter_early)
            {
              improvement_rate = (cost_early - cost_late) / (iter_late - iter_early) * 1000.0;
            }
          }
        }

        // Count iterations (approximation: max_iter if not converged)
        unsigned iterations = converged ? 0 : max_iter;  // Will need to track this properly in train()

        // Write results to file
        robust_file << depth << " " << width << " " << repeat << " "
                    << final_cost << " " << (converged ? 1 : 0) << " "
                    << iterations << " " << improvement_rate << std::endl;

        std::cout << "    Final cost: " << final_cost
                  << (converged ? " (CONVERGED)" : " (did not converge)") << std::endl;
      }
      std::cout << std::endl;
    }
  }

  robust_file.close();

  std::cout << "Analysis complete" << std::endl;

  // Cleanup
  delete activation_function_pt;
  activation_function_pt = 0;

  return 0;
}
