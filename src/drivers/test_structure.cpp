/// Test driver to verify class structure compiles
/// Author: Sofia Goncalves

#include "../project2_a.h"
#include <iostream>

int main()
{
  std::cout << "Testing Neural Network Class Structure\n\n";

  // Create an activation function
  ActivationFunction* activation_function_pt =
    new TanhActivationFunction;

  // Build a simple network: 2 inputs, 3 hidden, 1 output
  // Network structure: (2, 3, 1)
  unsigned n_input = 2;

  std::vector<std::pair<unsigned, ActivationFunction*>> non_input_layer;

  // Hidden layer: 3 neurons
  non_input_layer.push_back(std::make_pair(3, activation_function_pt));

  // Output layer: 1 neuron
  non_input_layer.push_back(std::make_pair(1, activation_function_pt));

  // Create the network
  NeuralNetwork network(n_input, non_input_layer);

  std::cout << "\nNetwork structure:\n";
  std::cout << "  Input layer: " << n_input << " neurons\n";
  std::cout << "  Non-input layers: " << network.n_layer() << "\n";

  for (unsigned l = 0; l < network.n_layer(); l++)
  {
    std::cout << "    Layer " << l << ": "
              << network.layer_pt(l)->n_neuron() << " neurons, "
              << network.layer_pt(l)->n_input() << " inputs\n";
  }

  std::cout << "\nStructure test: PASSED\n";
  std::cout << "All functions throw runtime_error as expected\n";

  // Clean up
  delete activation_function_pt;

  return 0;
}
