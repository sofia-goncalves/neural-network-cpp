/// Validation test for feed_forward implementation
/// Uses known weights/biases and manually computed expected output
/// Author: Sofia Goncalves

#include "../project2_a.h"
#include <iostream>
#include <cmath>
#include <iomanip>

int main()
{
  std::cout << "Validating Feed-Forward Implementation\n\n";

  // Create activation function
  ActivationFunction* tanh_pt = new TanhActivationFunction;

  // Build network with structure: (2, 4, 4, 1)
  // Input: 2, Hidden layer 0: 4 neurons, Hidden layer 1: 4 neurons, Output: 1
  unsigned n_input = 2;
  std::vector<std::pair<unsigned, ActivationFunction*>> non_input_layer;
  non_input_layer.push_back(std::make_pair(4, tanh_pt));  // Hidden layer 0
  non_input_layer.push_back(std::make_pair(4, tanh_pt));  // Hidden layer 1
  non_input_layer.push_back(std::make_pair(1, tanh_pt));  // Output layer

  NeuralNetwork network(n_input, non_input_layer);

  std::cout << "Network structure: (2, 4, 4, 1)\n\n";

  // Set known weights and biases for validation
  // Layer 0: 4 neurons, 2 inputs
  // Use simple values for manual calculation
  DoubleMatrix& W0 = network.layer_pt(0)->weight();
  DoubleVector& b0 = network.layer_pt(0)->bias();

  // W0: Simple pattern for manual verification
  W0(0,0) = 1.0;  W0(0,1) = 0.0;
  W0(1,0) = 0.0;  W0(1,1) = 1.0;
  W0(2,0) = 0.5;  W0(2,1) = 0.5;
  W0(3,0) = -1.0; W0(3,1) = 1.0;

  b0[0] = 0.0;
  b0[1] = 0.0;
  b0[2] = 0.0;
  b0[3] = 0.0;

  // Layer 1: 4 neurons, 4 inputs
  DoubleMatrix& W1 = network.layer_pt(1)->weight();
  DoubleVector& b1 = network.layer_pt(1)->bias();

  // W1: Identity-like pattern
  W1(0,0) = 1.0;  W1(0,1) = 0.0;  W1(0,2) = 0.0;  W1(0,3) = 0.0;
  W1(1,0) = 0.0;  W1(1,1) = 1.0;  W1(1,2) = 0.0;  W1(1,3) = 0.0;
  W1(2,0) = 0.0;  W1(2,1) = 0.0;  W1(2,2) = 1.0;  W1(2,3) = 0.0;
  W1(3,0) = 0.0;  W1(3,1) = 0.0;  W1(3,2) = 0.0;  W1(3,3) = 1.0;

  b1[0] = 0.0;
  b1[1] = 0.0;
  b1[2] = 0.0;
  b1[3] = 0.0;

  // Layer 2 (output): 1 neuron, 4 inputs
  DoubleMatrix& W2 = network.layer_pt(2)->weight();
  DoubleVector& b2 = network.layer_pt(2)->bias();

  // W2: Average of inputs
  W2(0,0) = 0.25;
  W2(0,1) = 0.25;
  W2(0,2) = 0.25;
  W2(0,3) = 0.25;

  b2[0] = 0.0;

  // Test input
  DoubleVector input(2);
  input[0] = 1.0;
  input[1] = 2.0;

  std::cout << "Input: [" << input[0] << ", " << input[1] << "]\n\n";

  // Manual calculation for validation
  std::cout << "Manual calculation:\n";

  // Layer 0: z[0] = W0 * input + b0, a[0] = tanh(z[0])
  std::cout << "Layer 0:\n";
  double z0_0 = 1.0*1.0 + 0.0*2.0 + 0.0;  // = 1.0
  double z0_1 = 0.0*1.0 + 1.0*2.0 + 0.0;  // = 2.0
  double z0_2 = 0.5*1.0 + 0.5*2.0 + 0.0;  // = 1.5
  double z0_3 = -1.0*1.0 + 1.0*2.0 + 0.0; // = 1.0

  double a0_0 = std::tanh(z0_0);
  double a0_1 = std::tanh(z0_1);
  double a0_2 = std::tanh(z0_2);
  double a0_3 = std::tanh(z0_3);

  std::cout << "  z[0] = [" << z0_0 << ", " << z0_1 << ", " << z0_2 << ", " << z0_3 << "]\n";
  std::cout << "  a[0] = [" << a0_0 << ", " << a0_1 << ", " << a0_2 << ", " << a0_3 << "]\n\n";

  // Layer 1: z[1] = W1 * a[0] + b1, a[1] = tanh(z[1])
  // Since W1 is identity, z[1] = a[0]
  std::cout << "Layer 1 (identity matrix):\n";
  double z1_0 = a0_0;
  double z1_1 = a0_1;
  double z1_2 = a0_2;
  double z1_3 = a0_3;

  double a1_0 = std::tanh(z1_0);
  double a1_1 = std::tanh(z1_1);
  double a1_2 = std::tanh(z1_2);
  double a1_3 = std::tanh(z1_3);

  std::cout << "  z[1] = [" << z1_0 << ", " << z1_1 << ", " << z1_2 << ", " << z1_3 << "]\n";
  std::cout << "  a[1] = [" << a1_0 << ", " << a1_1 << ", " << a1_2 << ", " << a1_3 << "]\n\n";

  // Layer 2: z[2] = W2 * a[1] + b2, a[2] = tanh(z[2])
  std::cout << "Layer 2 (output):\n";
  double z2_0 = 0.25 * (a1_0 + a1_1 + a1_2 + a1_3);
  double a2_0 = std::tanh(z2_0);

  std::cout << "  z[2] = [" << z2_0 << "]\n";
  std::cout << "  a[2] = [" << a2_0 << "]\n\n";

  std::cout << "Expected output: " << a2_0 << "\n\n";

  // Now run feed_forward and compare
  DoubleVector output;
  network.feed_forward(input, output);

  std::cout << "Network output:  " << output[0] << "\n\n";

  // Check if they match (within numerical tolerance)
  double tolerance = 1.0e-14;
  double error = std::abs(output[0] - a2_0);

  std::cout << "Absolute error:  " << error << "\n";

  if (error < tolerance)
  {
    std::cout << "\nValidation test: PASSED\n";
  }
  else
  {
    std::cout << "\nValidation test: FAILED\n";
    std::cout << "Error exceeds tolerance of " << tolerance << "\n";
  }

  // Clean up
  delete tanh_pt;

  return (error < tolerance) ? 0 : 1;
}
