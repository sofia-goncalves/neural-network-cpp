#ifndef PROJECT2_A_H
#define PROJECT2_A_H

/// Neural Network Implementation for Binary Classification
/// Author: Sofia Goncalves
/// Student ID: 11058869

#include "project2_a_basics.h"
#include <stdexcept>
#include <iostream>

/// Forward declaration of NeuralNetworkLayer
class NeuralNetworkLayer;


/// Neural Network Layer Class
/// Represents a single layer in the network (non-input layers only)
/// Design: Encapsulates all data and operations for one layer (weights, biases, activation)
class NeuralNetworkLayer
{
private:
  /// Number of neurons in this layer
  unsigned N_neuron;

  /// Number of inputs to this layer (i.e. neurons in previous layer)
  unsigned N_input;

  /// Weight matrix W[l] (N_neuron × N_input)
  /// W(i,j) = weight from neuron j in previous layer to neuron i in this layer
  DoubleMatrix Weight;

  /// Bias vector b[l] (N_neuron)
  DoubleVector Bias;

  /// Pointer to activation function for this layer
  /// Design: Stored not owned - allows sharing activation functions between layers
  ActivationFunction* Activation_function_pt;

  /// Storage for intermediate values (used during feed-forward and back-propagation)
  /// Design: Stored to avoid recomputation during gradient calculation
  /// z[l] = input to neurons (before activation)
  DoubleVector Z;

  /// a[l] = output from neurons (after activation)
  DoubleVector A;

public:
  /// Constructor
  NeuralNetworkLayer(const unsigned& n_input,
                     const unsigned& n_neuron,
                     ActivationFunction* activation_function_pt)
    : N_neuron(n_neuron),
      N_input(n_input),
      Weight(n_neuron, n_input),
      Bias(n_neuron),
      Activation_function_pt(activation_function_pt),
      Z(n_neuron),
      A(n_neuron)
  {
    // Weights and biases initialized to zero by default
    // Will be set properly by initialise_parameters() or read_parameters_from_disk()
  }

  /// Compute output from this layer given input from previous layer
  /// Implements: z[l] = W[l] * a[l-1] + b[l]
  ///             a[l] = σ(z[l])
  void feed_forward(const DoubleVector& input, DoubleVector& output)
  {
    // Compute z = W * input + b
    // For each neuron i: z[i] = sum_j(W[i][j] * input[j]) + b[i]
    for (unsigned i = 0; i < N_neuron; i++)
    {
      Z[i] = Bias[i];  // Start with bias
      for (unsigned j = 0; j < N_input; j++)
      {
        Z[i] += Weight(i, j) * input[j];
      }
    }

    // Apply activation function: a = σ(z)
    for (unsigned i = 0; i < N_neuron; i++)
    {
      A[i] = Activation_function_pt->sigma(Z[i]);
    }

    // Copy to output
    output.resize(N_neuron);
    for (unsigned i = 0; i < N_neuron; i++)
    {
      output[i] = A[i];
    }
  }

  /// Access functions (for training and file I/O)
  DoubleMatrix& weight() { return Weight; }
  const DoubleMatrix& weight() const { return Weight; }

  DoubleVector& bias() { return Bias; }
  const DoubleVector& bias() const { return Bias; }

  ActivationFunction* activation_function_pt() { return Activation_function_pt; }
  const ActivationFunction* activation_function_pt() const { return Activation_function_pt; }

  const DoubleVector& z() const { return Z; }
  const DoubleVector& a() const { return A; }

  unsigned n_neuron() const { return N_neuron; }
  unsigned n_input() const { return N_input; }
};


/// Neural Network Class
/// Fully-connected feedforward neural network
/// Design: Composition pattern - network comprises multiple layer objects
class NeuralNetwork : public NeuralNetworkBasis
{
private:
  /// Number of inputs (neurons in input layer)
  unsigned N_input;

  /// Storage for the non-input layers (hidden + output)
  /// Design: Pointers allow explicit memory management and future polymorphism
  /// Layer[0] = first hidden layer
  /// Layer[L-2] = last hidden layer
  /// Layer[L-1] = output layer
  std::vector<NeuralNetworkLayer*> Layer_pt;

public:
  /// Constructor: Pass the number of inputs (i.e. the number of
  /// neurons in the input layer), n_input, and a vector of pairs,
  /// containing for each subsequent layer (incl. the output layer)
  /// (i) the number of neurons in that layer
  /// (ii) a pointer to the activation function to be used by
  ///      all neurons in that layer
  NeuralNetwork(
    const unsigned& n_input,
    const std::vector<std::pair<unsigned, ActivationFunction*>>&
      non_input_layer)
    : N_input(n_input)
  {
    // Number of non-input layers
    unsigned n_layer = non_input_layer.size();

    // Create each layer
    for (unsigned l = 0; l < n_layer; l++)
    {
      // Number of inputs to this layer = number of neurons in previous layer
      unsigned n_input_to_layer = (l == 0) ? N_input : non_input_layer[l-1].first;

      // Number of neurons in this layer
      unsigned n_neuron = non_input_layer[l].first;

      // Activation function for this layer
      ActivationFunction* act_fn_pt = non_input_layer[l].second;

      // Create and store the layer
      Layer_pt.push_back(new NeuralNetworkLayer(n_input_to_layer,
                                                  n_neuron,
                                                  act_fn_pt));
    }

    // Network created (silent - only output if needed for debugging)
  }

  /// Destructor: Clean up allocated memory
  ~NeuralNetwork()
  {
    unsigned n_layer = Layer_pt.size();
    for (unsigned l = 0; l < n_layer; l++)
    {
      delete Layer_pt[l];
    }
  }

  /// Number of layers (non-input)
  unsigned n_layer() const { return Layer_pt.size(); }

  /// Access to specific layer
  NeuralNetworkLayer* layer_pt(const unsigned& l) { return Layer_pt[l]; }
  const NeuralNetworkLayer* layer_pt(const unsigned& l) const { return Layer_pt[l]; }

  // Pure virtual functions that must be implemented

  /// Feed-forward algorithm: compute output from input
  /// Implements equations (7) and (8) from project description
  void feed_forward(const DoubleVector& input,
                    DoubleVector& output) const
  {
    // a[1] = x (equation 7)
    DoubleVector current_input = input;

    // Feed through each layer: a[l] = σ(W[l]*a[l-1] + b[l]) (equation 8)
    unsigned n_layer = Layer_pt.size();
    for (unsigned l = 0; l < n_layer; l++)
    {
      DoubleVector layer_output;
      Layer_pt[l]->feed_forward(current_input, layer_output);
      current_input = layer_output;
    }

    // Output is from final layer: a[L]
    output = current_input;
  }

  /// Get cost for single input/target pair
  /// Implements equation (19): C = (1/2)||y - a[L]||^2
  double cost(const DoubleVector& input,
              const DoubleVector& target_output) const
  {
    // Get network output a[L] for this input
    DoubleVector output;
    feed_forward(input, output);

    // Compute error: e = y - a[L]
    // Then compute cost: C = (1/2) * sum(e_i^2)
    double cost_value = 0.0;
    unsigned n_output = output.n();
    for (unsigned i = 0; i < n_output; i++)
    {
      double error = target_output[i] - output[i];
      cost_value += error * error;
    }
    cost_value *= 0.5;

    return cost_value;
  }

  /// Get cost for training data
  /// Implements equation (9): C = (1/N) * sum(C_i)
  double cost_for_training_data(
    const std::vector<std::pair<DoubleVector,DoubleVector>> training_data)
    const
  {
    unsigned n_data = training_data.size();
    double total_cost = 0.0;

    // Sum cost over all training examples
    for (unsigned i = 0; i < n_data; i++)
    {
      const DoubleVector& input = training_data[i].first;
      const DoubleVector& target = training_data[i].second;
      total_cost += cost(input, target);
    }

    // Average cost
    return total_cost / n_data;
  }

  /// Write parameters to disk
  void write_parameters_to_disk(const std::string& filename) const
  {
    std::string error_msg =
      "NeuralNetwork::write_parameters_to_disk() not yet implemented\n";
    throw std::runtime_error(error_msg);
  }

  /// Read parameters from disk
  void read_parameters_from_disk(const std::string& filename)
  {
    std::string error_msg =
      "NeuralNetwork::read_parameters_from_disk() not yet implemented\n";
    throw std::runtime_error(error_msg);
  }

  /// Train the network
  void train(
    const std::vector<std::pair<DoubleVector,DoubleVector>>& training_data,
    const double& learning_rate,
    const double& tol_training,
    const unsigned& max_iter,
    const std::string& convergence_history_file_name="")
  {
    std::string error_msg =
      "NeuralNetwork::train() not yet implemented\n";
    throw std::runtime_error(error_msg);
  }

  /// Initialise parameters with random values from normal distribution
  void initialise_parameters(const double& mean, const double& std_dev)
  {
    // Set up normal distribution
    std::normal_distribution<> normal_dist(mean, std_dev);

    // Loop over all layers
    unsigned n_layer = Layer_pt.size();
    for (unsigned l = 0; l < n_layer; l++)
    {
      // Get weight matrix and bias vector for this layer
      DoubleMatrix& W = Layer_pt[l]->weight();
      DoubleVector& b = Layer_pt[l]->bias();

      // Initialise weights
      for (unsigned i = 0; i < W.n(); i++)
      {
        for (unsigned j = 0; j < W.m(); j++)
        {
          W(i,j) = normal_dist(RandomNumber::Random_number_generator);
        }
      }

      // Initialise biases
      for (unsigned i = 0; i < b.n(); i++)
      {
        b[i] = normal_dist(RandomNumber::Random_number_generator);
      }
    }
  }
};

#endif // PROJECT2_A_H
