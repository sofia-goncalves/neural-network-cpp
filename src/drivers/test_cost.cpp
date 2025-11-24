/// Validation test for cost function implementation
/// Uses known outputs and targets to verify cost calculation
/// Author: Sofia Goncalves

#include "../project2_a.h"
#include <iostream>
#include <cmath>

int main()
{
  std::cout << "Validating Cost Function Implementation\n\n";

  // Create a simple network: (2, 1)
  // Just input layer (2) -> output layer (1)
  ActivationFunction* tanh_pt = new TanhActivationFunction;

  unsigned n_input = 2;
  std::vector<std::pair<unsigned, ActivationFunction*>> non_input_layer;
  non_input_layer.push_back(std::make_pair(1, tanh_pt));

  NeuralNetwork network(n_input, non_input_layer);

  // Set known weights and bias for predictable output
  DoubleMatrix& W = network.layer_pt(0)->weight();
  DoubleVector& b = network.layer_pt(0)->bias();

  W(0,0) = 1.0;
  W(0,1) = 0.0;
  b[0] = 0.0;

  std::cout << "Network: (2, 1)\n";
  std::cout << "W = [1.0, 0.0], b = [0.0]\n";
  std::cout << "Output will be: a = tanh(1.0 * x[0] + 0.0 * x[1] + 0.0) = tanh(x[0])\n\n";

  // Test 1: Single cost calculation
  std::cout << "Test 1: Single cost calculation\n";

  DoubleVector input(2);
  input[0] = 0.5;
  input[1] = 0.0;  // Ignored due to weight = 0

  DoubleVector target(1);
  target[0] = 0.8;

  // Manual calculation:
  // output = tanh(0.5) = 0.46211715726
  // error = 0.8 - 0.46211715726 = 0.33788284274
  // cost = 0.5 * (0.33788284274)^2 = 0.05707047879

  double output_manual = std::tanh(0.5);
  double error_manual = 0.8 - output_manual;
  double cost_manual = 0.5 * error_manual * error_manual;

  std::cout << "  Input: [" << input[0] << ", " << input[1] << "]\n";
  std::cout << "  Target: [" << target[0] << "]\n";
  std::cout << "  Expected output: " << output_manual << "\n";
  std::cout << "  Expected error: " << error_manual << "\n";
  std::cout << "  Expected cost: " << cost_manual << "\n\n";

  double cost_computed = network.cost(input, target);
  std::cout << "  Computed cost: " << cost_computed << "\n";

  double cost_error = std::abs(cost_computed - cost_manual);
  std::cout << "  Absolute error: " << cost_error << "\n";

  bool test1_passed = (cost_error < 1.0e-14);
  std::cout << "  Test 1: " << (test1_passed ? "PASSED" : "FAILED") << "\n\n";

  // Test 2: Cost for training data
  std::cout << "Test 2: Cost for training data (multiple samples)\n";

  std::vector<std::pair<DoubleVector,DoubleVector>> training_data;

  // Sample 1
  DoubleVector in1(2);
  in1[0] = 0.5; in1[1] = 0.0;
  DoubleVector tgt1(1);
  tgt1[0] = 0.8;
  training_data.push_back(std::make_pair(in1, tgt1));

  // Sample 2
  DoubleVector in2(2);
  in2[0] = -0.3; in2[1] = 0.0;
  DoubleVector tgt2(1);
  tgt2[0] = -0.5;
  training_data.push_back(std::make_pair(in2, tgt2));

  // Sample 3
  DoubleVector in3(2);
  in3[0] = 0.0; in3[1] = 0.0;
  DoubleVector tgt3(1);
  tgt3[0] = 0.1;
  training_data.push_back(std::make_pair(in3, tgt3));

  // Manual calculation of average cost
  double cost1 = network.cost(in1, tgt1);
  double cost2 = network.cost(in2, tgt2);
  double cost3 = network.cost(in3, tgt3);
  double avg_cost_manual = (cost1 + cost2 + cost3) / 3.0;

  std::cout << "  3 training samples\n";
  std::cout << "  Individual costs: [" << cost1 << ", " << cost2 << ", " << cost3 << "]\n";
  std::cout << "  Expected average: " << avg_cost_manual << "\n";

  double avg_cost_computed = network.cost_for_training_data(training_data);
  std::cout << "  Computed average: " << avg_cost_computed << "\n";

  double avg_error = std::abs(avg_cost_computed - avg_cost_manual);
  std::cout << "  Absolute error: " << avg_error << "\n";

  bool test2_passed = (avg_error < 1.0e-14);
  std::cout << "  Test 2: " << (test2_passed ? "PASSED" : "FAILED") << "\n\n";

  // Overall result
  bool all_passed = test1_passed && test2_passed;
  std::cout << (all_passed ? "All tests: PASSED\n" : "Some tests: FAILED\n");

  delete tanh_pt;

  return all_passed ? 0 : 1;
}
