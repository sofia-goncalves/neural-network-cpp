/// Validation test for file I/O implementation
/// Tests reading provided test data file and write/read consistency

#include "../project2_a.h"
#include <iostream>
#include <cmath>

int main()
{
  std::cout << "Validating File I/O Implementation\n\n";

  // Test 1: Read provided test data file
  std::cout << "Test 1: Reading provided test data\n";

  ActivationFunction* tanh_pt = new TanhActivationFunction;

  // Create network matching provided file structure: (2, 3, 3, 1)
  unsigned n_input = 2;
  std::vector<std::pair<unsigned, ActivationFunction*>> non_input_layer;
  non_input_layer.push_back(std::make_pair(3, tanh_pt));
  non_input_layer.push_back(std::make_pair(3, tanh_pt));
  non_input_layer.push_back(std::make_pair(1, tanh_pt));

  NeuralNetwork network1(n_input, non_input_layer);

  std::cout << "Network structure: (2, 3, 3, 1)\n";

  // Read provided test data
  std::string provided_file = "data/input/project_test_data.dat";
  std::cout << "Reading from: " << provided_file << "\n";

  network1.read_parameters_from_disk(provided_file);
  std::cout << "Read successful\n\n";

  // Test with an input
  DoubleVector input(2);
  input[0] = 0.5;
  input[1] = 1.0;

  DoubleVector output1;
  network1.feed_forward(input, output1);

  std::cout << "Network output for [0.5, 1.0]: " << output1[0] << "\n\n";

  // Test 2: Write and read back (round-trip test)
  std::cout << "Test 2: Write/Read round-trip\n";

  std::string output_file = "data/output/test_parameters.dat";
  std::cout << "Writing to: " << output_file << "\n";
  network1.write_parameters_to_disk(output_file);
  std::cout << "Write successful\n";

  // Create second network with same structure
  NeuralNetwork network2(n_input, non_input_layer);

  // Initialise with random values
  network2.initialise_parameters(0.0, 1.0);

  DoubleVector output2_before;
  network2.feed_forward(input, output2_before);
  std::cout << "Network2 output before read: " << output2_before[0] << "\n";

  // Read from file
  std::cout << "Reading back from: " << output_file << "\n";
  network2.read_parameters_from_disk(output_file);
  std::cout << "Read successful\n";

  DoubleVector output2_after;
  network2.feed_forward(input, output2_after);
  std::cout << "Network2 output after read: " << output2_after[0] << "\n\n";

  // Verify they match
  double error = std::abs(output1[0] - output2_after[0]);
  std::cout << "Absolute error: " << error << "\n";

  double tolerance = 1.0e-14;
  bool test_passed = (error < tolerance);

  if (test_passed)
  {
    std::cout << "\nFile I/O test: PASSED\n";
    std::cout << "Read/write round-trip successful\n";
  }
  else
  {
    std::cout << "\nFile I/O test: FAILED\n";
    std::cout << "Error " << error << " exceeds tolerance " << tolerance << "\n";
  }

  delete tanh_pt;

  return test_passed ? 0 : 1;
}
