/// Test driver to validate back-propagation against finite-differencing
/// Compares gradients computed by both methods for a simple network
/// Following proper validation procedure from project instructions

#include <iostream>
#include <cmath>
#include "../project2_a.h"

int main()
{
  std::cout << "Gradient Validation Test" << std::endl;
  std::cout << std::endl;

  // Create activation function
  ActivationFunction* activation_function_pt = new TanhActivationFunction;

  // Build a small network: (2, 3, 1) for easy testing
  unsigned n_input = 2;
  std::vector<std::pair<unsigned, ActivationFunction*>> non_input_layer;

  // First hidden layer: 3 neurons
  non_input_layer.push_back(std::make_pair(3, activation_function_pt));

  // Output layer: 1 neuron
  non_input_layer.push_back(std::make_pair(1, activation_function_pt));

  NeuralNetwork network(n_input, non_input_layer);

  // Initialise with small random values
  network.initialise_parameters(0.0, 0.5);

  // Create a test input and target
  DoubleVector input(2);
  input[0] = 0.3;
  input[1] = 0.7;

  DoubleVector target(1);
  target[0] = 1.0;

  std::cout << "Network: (2, 3, 1)" << std::endl;
  std::cout << "Input: [" << input[0] << ", " << input[1] << "]" << std::endl;
  std::cout << "Target: [" << target[0] << "]" << std::endl;
  std::cout << std::endl;

  // Compute gradients via finite-difference
  std::cout << "Computing gradients via finite-difference" << std::endl;
  std::vector<DoubleMatrix> grad_W_fd;
  std::vector<DoubleVector> grad_b_fd;

  double epsilon = 1.0e-7;
  network.test_compute_gradients_finite_difference(input, target,
                                                    grad_W_fd, grad_b_fd,
                                                    epsilon);
  std::cout << "done" << std::endl;

  // Compute gradients via back-propagation
  std::cout << "Computing gradients via back-propagation" << std::endl;
  std::vector<DoubleMatrix> grad_W_bp;
  std::vector<DoubleVector> grad_b_bp;

  network.test_compute_gradients_backprop(input, target, grad_W_bp, grad_b_bp);
  std::cout << "done" << std::endl;
  std::cout << std::endl;

  // Compare gradients
  std::cout << "Comparing gradients:" << std::endl;
  std::cout << std::endl;

  bool all_match = true;
  double max_rel_error = 0.0;
  double tolerance = 1.0e-5;  // Relative error tolerance

  unsigned n_layer = grad_W_fd.size();

  for (unsigned l = 0; l < n_layer; l++)
  {
    std::cout << "Layer " << l << ":" << std::endl;

    // Compare bias gradients
    std::cout << "  Bias gradients:" << std::endl;
    for (unsigned j = 0; j < grad_b_fd[l].n(); j++)
    {
      double fd_val = grad_b_fd[l][j];
      double bp_val = grad_b_bp[l][j];
      double abs_diff = std::abs(fd_val - bp_val);
      double rel_error = abs_diff / (std::abs(fd_val) + 1.0e-10);

      std::cout << "    b[" << l << "][" << j << "]: "
                << "FD = " << fd_val << ", "
                << "BP = " << bp_val << ", "
                << "rel_error = " << rel_error;

      if (rel_error > tolerance)
      {
        std::cout << " MISMATCH";
        all_match = false;
      }
      std::cout << std::endl;

      if (rel_error > max_rel_error)
      {
        max_rel_error = rel_error;
      }
    }

    // Compare weight gradients (show only a few for brevity)
    std::cout << "  Weight gradients (sample):" << std::endl;
    unsigned n_show = (grad_W_fd[l].n() < 3) ? grad_W_fd[l].n() : 3;
    unsigned m_show = (grad_W_fd[l].m() < 3) ? grad_W_fd[l].m() : 3;

    for (unsigned j = 0; j < n_show; j++)
    {
      for (unsigned k = 0; k < m_show; k++)
      {
        double fd_val = grad_W_fd[l](j, k);
        double bp_val = grad_W_bp[l](j, k);
        double abs_diff = std::abs(fd_val - bp_val);
        double rel_error = abs_diff / (std::abs(fd_val) + 1.0e-10);

        std::cout << "    W[" << l << "](" << j << "," << k << "): "
                  << "FD = " << fd_val << ", "
                  << "BP = " << bp_val << ", "
                  << "rel_error = " << rel_error;

        if (rel_error > tolerance)
        {
          std::cout << " MISMATCH";
          all_match = false;
        }
        std::cout << std::endl;

        if (rel_error > max_rel_error)
        {
          max_rel_error = rel_error;
        }
      }
    }

    // Check all weights (not just displayed ones)
    for (unsigned j = 0; j < grad_W_fd[l].n(); j++)
    {
      for (unsigned k = 0; k < grad_W_fd[l].m(); k++)
      {
        double fd_val = grad_W_fd[l](j, k);
        double bp_val = grad_W_bp[l](j, k);
        double abs_diff = std::abs(fd_val - bp_val);
        double rel_error = abs_diff / (std::abs(fd_val) + 1.0e-10);

        if (rel_error > tolerance)
        {
          all_match = false;
        }

        if (rel_error > max_rel_error)
        {
          max_rel_error = rel_error;
        }
      }
    }

    std::cout << std::endl;
  }

  // Summary
  std::cout << "Summary:" << std::endl;
  std::cout << "  Maximum relative error: " << max_rel_error << std::endl;
  std::cout << "  Tolerance: " << tolerance << std::endl;

  if (all_match)
  {
    std::cout << "  Status: PASS" << std::endl;
    std::cout << std::endl;
    std::cout << "Back-propagation gradients match finite-difference" << std::endl;
  }
  else
  {
    std::cout << "  Status: FAIL" << std::endl;
    std::cout << std::endl;
    std::cout << "ERROR: Back-propagation gradients do not match" << std::endl;
  }

  // Clean up
  delete activation_function_pt;
  activation_function_pt = 0;

  return all_match ? 0 : 1;
}
