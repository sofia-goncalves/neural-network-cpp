#ifndef PROJECT2_A_H
#define PROJECT2_A_H

/// Neural Network Implementation for Binary Classification

#include "project2_a_basics.h"
#include <stdexcept>
#include <iostream>
#include <fstream>

// Debug output control - comment out to disable debug messages
// #define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "[DEBUG] " << msg << std::endl
#else
#define DEBUG_PRINT(msg)
#endif

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
    // Weights and biases initialised to zero by default
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
      Layer_pt[l] = 0;
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
  /// File format (matches provided test data):
  /// For each layer:
  ///   ActivationFunctionName
  ///   n_inputs_to_layer
  ///   n_neurons_in_layer
  ///   neuron_index bias_value (for each neuron)
  ///   neuron_index input_index weight_value (for each weight)
  void write_parameters_to_disk(const std::string& filename) const
  {
    std::ofstream outfile(filename.c_str());
    if (!outfile)
    {
      throw std::runtime_error("Cannot open file: " + filename);
    }

    unsigned n_layer = Layer_pt.size();

    // Write parameters for each layer
    for (unsigned l = 0; l < n_layer; l++)
    {
      const DoubleMatrix& W = Layer_pt[l]->weight();
      const DoubleVector& b = Layer_pt[l]->bias();

      // Write activation function name
      outfile << Layer_pt[l]->activation_function_pt()->name() << std::endl;

      // Write layer structure
      outfile << Layer_pt[l]->n_input() << std::endl;
      outfile << Layer_pt[l]->n_neuron() << std::endl;

      // Write biases with indices
      for (unsigned i = 0; i < b.n(); i++)
      {
        outfile << i << " " << b[i] << " " << std::endl;
      }

      // Write weights with indices
      for (unsigned i = 0; i < W.n(); i++)
      {
        for (unsigned j = 0; j < W.m(); j++)
        {
          outfile << i << " " << j << " " << W(i,j) << " " << std::endl;
        }
      }
    }

    outfile.close();
  }

  /// Read parameters from disk
  /// Reads file in provided format with sanity checks
  /// Validates network structure matches before loading parameters
  void read_parameters_from_disk(const std::string& filename)
  {
    std::ifstream infile(filename.c_str());
    if (!infile)
    {
      throw std::runtime_error("Cannot open file: " + filename);
    }

    unsigned n_layer = Layer_pt.size();

    // Read parameters for each layer
    for (unsigned l = 0; l < n_layer; l++)
    {
      DoubleMatrix& W = Layer_pt[l]->weight();
      DoubleVector& b = Layer_pt[l]->bias();

      // Read and validate activation function name
      std::string activation_name;
      infile >> activation_name;
      if (activation_name != Layer_pt[l]->activation_function_pt()->name())
      {
        throw std::runtime_error("Layer " + std::to_string(l) +
                                 " activation function mismatch: expected " +
                                 Layer_pt[l]->activation_function_pt()->name() +
                                 ", got " + activation_name);
      }

      // Read and validate layer structure
      unsigned n_input_file, n_neuron_file;
      infile >> n_input_file >> n_neuron_file;

      if (n_input_file != Layer_pt[l]->n_input())
      {
        throw std::runtime_error("Layer " + std::to_string(l) +
                                 " n_input mismatch: expected " +
                                 std::to_string(Layer_pt[l]->n_input()) +
                                 ", got " + std::to_string(n_input_file));
      }

      if (n_neuron_file != Layer_pt[l]->n_neuron())
      {
        throw std::runtime_error("Layer " + std::to_string(l) +
                                 " n_neuron mismatch: expected " +
                                 std::to_string(Layer_pt[l]->n_neuron()) +
                                 ", got " + std::to_string(n_neuron_file));
      }

      // Read biases with index validation
      for (unsigned i = 0; i < n_neuron_file; i++)
      {
        unsigned idx;
        double bias_value;
        infile >> idx >> bias_value;

        if (idx != i)
        {
          throw std::runtime_error("Layer " + std::to_string(l) +
                                   " bias index mismatch at position " + std::to_string(i));
        }

        b[idx] = bias_value;
      }

      // Read weights with index validation
      for (unsigned i = 0; i < n_neuron_file; i++)
      {
        for (unsigned j = 0; j < n_input_file; j++)
        {
          unsigned row_idx, col_idx;
          double weight_value;
          infile >> row_idx >> col_idx >> weight_value;

          if (row_idx != i || col_idx != j)
          {
            throw std::runtime_error("Layer " + std::to_string(l) +
                                     " weight index mismatch: expected (" +
                                     std::to_string(i) + "," + std::to_string(j) +
                                     "), got (" + std::to_string(row_idx) + "," +
                                     std::to_string(col_idx) + ")");
          }

          W(row_idx, col_idx) = weight_value;
        }
      }
    }

    infile.close();
  }

private:
  /// Compute gradients via finite-differencing for a single training point
  /// Used for validation of back-propagation implementation
  /// Returns gradients without updating parameters
  void compute_gradients_finite_difference(
    const DoubleVector& input,
    const DoubleVector& target,
    std::vector<DoubleMatrix>& grad_W,
    std::vector<DoubleVector>& grad_b,
    const double& epsilon = 1.0e-8)
  {
    unsigned n_layer = Layer_pt.size();

    // Clear and initialise gradient storage
    grad_W.clear();
    grad_b.clear();

    for (unsigned l = 0; l < n_layer; l++)
    {
      const DoubleMatrix& W = Layer_pt[l]->weight();
      const DoubleVector& b = Layer_pt[l]->bias();

      grad_W.push_back(DoubleMatrix(W.n(), W.m()));
      grad_b.push_back(DoubleVector(b.n()));
    }

    // Compute cost at current parameters
    double C0 = cost(input, target);

    // Compute gradient of biases via finite-difference
    for (unsigned l = 0; l < n_layer; l++)
    {
      DoubleVector& b = Layer_pt[l]->bias();

      for (unsigned j = 0; j < b.n(); j++)
      {
        // Perturb b[l]_j by +epsilon
        double original_value = b[j];
        b[j] = original_value + epsilon;

        // Compute perturbed cost
        double C_plus = cost(input, target);

        // Restore original value
        b[j] = original_value;

        // Finite-difference approximation: ∂C/∂b ≈ (C(b+ε) - C(b)) / ε
        grad_b[l][j] = (C_plus - C0) / epsilon;
      }
    }

    // Compute gradient of weights via finite-difference
    for (unsigned l = 0; l < n_layer; l++)
    {
      DoubleMatrix& W = Layer_pt[l]->weight();

      for (unsigned j = 0; j < W.n(); j++)
      {
        for (unsigned k = 0; k < W.m(); k++)
        {
          // Perturb w[l]_jk by +epsilon
          double original_value = W(j, k);
          W(j, k) = original_value + epsilon;

          // Compute perturbed cost
          double C_plus = cost(input, target);

          // Restore original value
          W(j, k) = original_value;

          // Finite-difference approximation: ∂C/∂w ≈ (C(w+ε) - C(w)) / ε
          grad_W[l](j, k) = (C_plus - C0) / epsilon;
        }
      }
    }
  }

  /// Compute gradients via back-propagation for a single training point
  /// Returns gradients without updating parameters (for validation)
  void compute_gradients_backprop(
    const DoubleVector& input,
    const DoubleVector& target,
    std::vector<DoubleMatrix>& grad_W,
    std::vector<DoubleVector>& grad_b)
  {
    unsigned n_layer = Layer_pt.size();

    // Step 1: Feed-forward to compute and store z[l] and a[l] for all layers
    DoubleVector current_input = input;
    for (unsigned l = 0; l < n_layer; l++)
    {
      DoubleVector layer_output;
      Layer_pt[l]->feed_forward(current_input, layer_output);
      current_input = layer_output;
    }

    // Storage for errors δ[l] for each layer
    std::vector<DoubleVector> delta(n_layer);

    // Step 2: Compute error for output layer (layer L)
    // δ[L] = σ'(z[L]) ◦ (a[L] - y)
    unsigned L = n_layer - 1;
    const DoubleVector& z_L = Layer_pt[L]->z();
    const DoubleVector& a_L = Layer_pt[L]->a();
    unsigned n_output = a_L.n();

    delta[L].resize(n_output);
    for (unsigned j = 0; j < n_output; j++)
    {
      double sigma_prime = Layer_pt[L]->activation_function_pt()->dsigma(z_L[j]);
      delta[L][j] = sigma_prime * (a_L[j] - target[j]);
    }

    // Step 3: Back-propagate errors through hidden layers
    // δ[l] = σ'(z[l]) ◦ (W[l+1])^T δ[l+1]
    for (int l = L - 1; l >= 0; l--)
    {
      const DoubleVector& z_l = Layer_pt[l]->z();
      const DoubleMatrix& W_next = Layer_pt[l+1]->weight();
      unsigned n_neuron = z_l.n();

      delta[l].resize(n_neuron);

      for (unsigned j = 0; j < n_neuron; j++)
      {
        // Compute (W[l+1])^T δ[l+1] for neuron j
        // This is sum over k: W[l+1](k,j) * δ[l+1][k]
        double sum = 0.0;
        for (unsigned k = 0; k < delta[l+1].n(); k++)
        {
          sum += W_next(k, j) * delta[l+1][k];
        }

        double sigma_prime = Layer_pt[l]->activation_function_pt()->dsigma(z_l[j]);
        delta[l][j] = sigma_prime * sum;
      }
    }

    // Step 4: Compute gradients from errors
    // ∂C/∂b[l]_j = δ[l]_j
    // ∂C/∂w[l]_jk = δ[l]_j * a[l-1]_k

    // Clear and initialise gradient storage
    grad_W.clear();
    grad_b.clear();

    for (unsigned l = 0; l < n_layer; l++)
    {
      const DoubleMatrix& W = Layer_pt[l]->weight();
      const DoubleVector& b = Layer_pt[l]->bias();

      grad_W.push_back(DoubleMatrix(W.n(), W.m()));
      grad_b.push_back(DoubleVector(b.n()));

      // Get a[l-1] (output from previous layer, or input for first layer)
      DoubleVector a_prev;
      if (l == 0)
      {
        a_prev = input;
      }
      else
      {
        a_prev = Layer_pt[l-1]->a();
      }

      // Gradient of biases: ∂C/∂b[l]_j = δ[l]_j
      for (unsigned j = 0; j < b.n(); j++)
      {
        grad_b[l][j] = delta[l][j];
      }

      // Gradient of weights: ∂C/∂w[l]_jk = δ[l]_j * a[l-1]_k
      for (unsigned j = 0; j < W.n(); j++)
      {
        for (unsigned k = 0; k < W.m(); k++)
        {
          grad_W[l](j, k) = delta[l][j] * a_prev[k];
        }
      }
    }
  }

  /// Compute gradients via back-propagation for a single training point
  /// and update weights and biases
  /// Implements Algorithm 1 from project description
  void back_propagation_and_update(
    const DoubleVector& input,
    const DoubleVector& target,
    const double& learning_rate)
  {
    unsigned n_layer = Layer_pt.size();

    // Step 1: Feed-forward to compute and store z[l] and a[l] for all layers
    DoubleVector current_input = input;
    for (unsigned l = 0; l < n_layer; l++)
    {
      DoubleVector layer_output;
      Layer_pt[l]->feed_forward(current_input, layer_output);
      current_input = layer_output;
    }

    // Storage for errors δ[l] for each layer
    std::vector<DoubleVector> delta(n_layer);

    // Step 2: Compute error for output layer (layer L)
    // δ[L] = σ'(z[L]) ◦ (a[L] - y)
    unsigned L = n_layer - 1;
    const DoubleVector& z_L = Layer_pt[L]->z();
    const DoubleVector& a_L = Layer_pt[L]->a();
    unsigned n_output = a_L.n();

    delta[L].resize(n_output);
    for (unsigned j = 0; j < n_output; j++)
    {
      double sigma_prime = Layer_pt[L]->activation_function_pt()->dsigma(z_L[j]);
      delta[L][j] = sigma_prime * (a_L[j] - target[j]);
    }

    DEBUG_PRINT("Output layer error δ[" << L << "][0] = " << delta[L][0]);

    // Step 3: Back-propagate errors through hidden layers
    // δ[l] = σ'(z[l]) ◦ (W[l+1])^T δ[l+1]
    for (int l = L - 1; l >= 0; l--)
    {
      const DoubleVector& z_l = Layer_pt[l]->z();
      const DoubleMatrix& W_next = Layer_pt[l+1]->weight();
      unsigned n_neuron = z_l.n();

      delta[l].resize(n_neuron);

      for (unsigned j = 0; j < n_neuron; j++)
      {
        // Compute (W[l+1])^T δ[l+1] for neuron j
        // This is sum over k: W[l+1](k,j) * δ[l+1][k]
        double sum = 0.0;
        for (unsigned k = 0; k < delta[l+1].n(); k++)
        {
          sum += W_next(k, j) * delta[l+1][k];
        }

        double sigma_prime = Layer_pt[l]->activation_function_pt()->dsigma(z_l[j]);
        delta[l][j] = sigma_prime * sum;
      }

      DEBUG_PRINT("Layer " << l << " error δ[" << l << "][0] = " << delta[l][0]);
    }

    // Step 4: Update weights and biases using gradients
    // ∂C/∂b[l]_j = δ[l]_j
    // ∂C/∂w[l]_jk = δ[l]_j * a[l-1]_k
    for (unsigned l = 0; l < n_layer; l++)
    {
      DoubleMatrix& W = Layer_pt[l]->weight();
      DoubleVector& b = Layer_pt[l]->bias();

      // Get a[l-1] (output from previous layer, or input for first layer)
      DoubleVector a_prev;
      if (l == 0)
      {
        a_prev = input;
      }
      else
      {
        a_prev = Layer_pt[l-1]->a();
      }

      // Update biases: b[l]_j ← b[l]_j - η * δ[l]_j
      for (unsigned j = 0; j < b.n(); j++)
      {
        b[j] -= learning_rate * delta[l][j];
      }

      // Update weights: w[l]_jk ← w[l]_jk - η * δ[l]_j * a[l-1]_k
      for (unsigned j = 0; j < W.n(); j++)
      {
        for (unsigned k = 0; k < W.m(); k++)
        {
          W(j, k) -= learning_rate * delta[l][j] * a_prev[k];
        }
      }
    }
  }

public:
  /// Train the network using stochastic gradient descent with back-propagation
  /// Implements Algorithm 2 from project description
  void train(
    const std::vector<std::pair<DoubleVector,DoubleVector>>& training_data,
    const double& learning_rate,
    const double& tol_training,
    const unsigned& max_iter,
    const std::string& convergence_history_file_name="")
  {
    DEBUG_PRINT("Starting training: learning_rate=" << learning_rate
                << ", tol=" << tol_training << ", max_iter=" << max_iter);

    unsigned n_data = training_data.size();

    // Open convergence history file if specified
    std::ofstream history_file;
    if (!convergence_history_file_name.empty())
    {
      history_file.open(convergence_history_file_name.c_str());
      if (!history_file)
      {
        throw std::runtime_error("Cannot open convergence history file: " +
                                 convergence_history_file_name);
      }
      history_file << "# Iteration  TotalCost" << std::endl;
    }

    // Training loop
    unsigned iter = 0;
    double current_cost = cost_for_training_data(training_data);

    DEBUG_PRINT("Initial cost: " << current_cost);

    while (iter < max_iter)
    {
      // Select random training data point
      std::uniform_int_distribution<unsigned> uniform_dist(0, n_data - 1);
      unsigned i = uniform_dist(RandomNumber::Random_number_generator);

      const DoubleVector& input = training_data[i].first;
      const DoubleVector& target = training_data[i].second;

      DEBUG_PRINT("Iteration " << iter << ": selected data point " << i);

      // Compute gradients by back-propagation and update parameters
      back_propagation_and_update(input, target, learning_rate);

      iter++;

      // Check convergence every 1000 iterations
      if (iter % 1000 == 0)
      {
        current_cost = cost_for_training_data(training_data);

        DEBUG_PRINT("Iteration " << iter << ": cost = " << current_cost);

        if (!convergence_history_file_name.empty())
        {
          history_file << iter << " " << current_cost << std::endl;
        }

        if (current_cost < tol_training)
        {
          std::cout << "Training converged at iteration " << iter
                    << " with cost " << current_cost << std::endl;
          if (history_file.is_open())
          {
            history_file.close();
          }
          return;
        }
      }
    }

    // Training did not converge
    std::cout << "Training did not converge after " << max_iter
              << " iterations. Final cost: " << current_cost << std::endl;

    if (history_file.is_open())
    {
      history_file.close();
    }
  }

  /// Train the network using optimised stochastic gradient descent
  /// This method implements the interlaced back-propagation algorithm
  /// from Higham & Higham (2019), where weight/bias updates are interlaced with error back-propagation.
  void train_optimised(
    const std::vector<std::pair<DoubleVector,DoubleVector>>& training_data,
    const double& learning_rate,
    const double& tol_training,
    const unsigned& max_iter,
    const std::string& convergence_history_file_name="")
  {
    DEBUG_PRINT("Starting optimised training: learning_rate=" << learning_rate
                << ", tol=" << tol_training << ", max_iter=" << max_iter);

    unsigned n_data = training_data.size();
    unsigned n_layer = Layer_pt.size();
    unsigned L = n_layer - 1;

    // Open convergence history file if specified
    std::ofstream history_file;
    if (!convergence_history_file_name.empty())
    {
      history_file.open(convergence_history_file_name.c_str());
      if (!history_file)
      {
        throw std::runtime_error("Cannot open convergence history file: " +
                                 convergence_history_file_name);
      }
      history_file << "# Iteration  TotalCost" << std::endl;
    }

    // Training loop
    unsigned iter = 0;
    double current_cost = cost_for_training_data(training_data);

    DEBUG_PRINT("Initial cost: " << current_cost);

    // Storage for current and next layer errors (only need 2 at a time)
    DoubleVector delta_current;
    DoubleVector delta_next;

    while (iter < max_iter)
    {
      // Select random training data point
      std::uniform_int_distribution<unsigned> uniform_dist(0, n_data - 1);
      unsigned i = uniform_dist(RandomNumber::Random_number_generator);

      const DoubleVector& input = training_data[i].first;
      const DoubleVector& target = training_data[i].second;

      DEBUG_PRINT("Iteration " << iter << ": selected data point " << i);

      // Step 1: Feed-forward to compute and store z[l] and a[l] for all layers
      DoubleVector current_input = input;
      for (unsigned l = 0; l < n_layer; l++)
      {
        DoubleVector layer_output;
        Layer_pt[l]->feed_forward(current_input, layer_output);
        current_input = layer_output;
      }

      // Step 2: Compute error for output layer (layer L)
      // δ[L] = σ'(z[L]) ◦ (a[L] - y)
      const DoubleVector& z_L = Layer_pt[L]->z();
      const DoubleVector& a_L = Layer_pt[L]->a();
      unsigned n_output = a_L.n();

      delta_current.resize(n_output);
      for (unsigned j = 0; j < n_output; j++)
      {
        double sigma_prime = Layer_pt[L]->activation_function_pt()->dsigma(z_L[j]);
        delta_current[j] = sigma_prime * (a_L[j] - target[j]);
      }

      DEBUG_PRINT("Output layer error δ[" << L << "][0] = " << delta_current[0]);

      // Step 3: Interlaced backward pass - compute next delta, then update current layer
      // Loop backwards from output layer to first hidden layer
      for (int l = L; l >= 0; l--)
      {
        const DoubleMatrix& W = Layer_pt[l]->weight();
        DoubleVector& b = Layer_pt[l]->bias();

        // FIRST: Compute error for previous layer (before updating current weights!)
        // δ[l-1] = σ'(z[l-1]) ◦ (W[l])^T δ[l]
        // This must happen BEFORE we update W[l]
        if (l > 0)
        {
          const DoubleVector& z_prev = Layer_pt[l-1]->z();
          unsigned n_neuron_prev = z_prev.n();

          delta_next.resize(n_neuron_prev);

          for (unsigned j = 0; j < n_neuron_prev; j++)
          {
            // Compute (W[l])^T δ[l] for neuron j in layer l-1
            // This is sum over k: W[l](k,j) * δ[l][k]
            double sum = 0.0;
            for (unsigned k = 0; k < delta_current.n(); k++)
            {
              sum += W(k, j) * delta_current[k];
            }

            double sigma_prime = Layer_pt[l-1]->activation_function_pt()->dsigma(z_prev[j]);
            delta_next[j] = sigma_prime * sum;
          }

          DEBUG_PRINT("Layer " << (l-1) << " error δ[" << (l-1) << "][0] = " << delta_next[0]);
        }

        // THEN: Update current layer weights and biases
        // Now we can safely modify W[l] and b[l]

        // Get a[l-1] (output from previous layer, or input for first layer)
        DoubleVector a_prev;
        if (l == 0)
        {
          a_prev = input;
        }
        else
        {
          a_prev = Layer_pt[l-1]->a();
        }

        // Need non-const reference to update weights
        DoubleMatrix& W_ref = Layer_pt[l]->weight();

        // Update biases: b[l]_j ← b[l]_j - η * δ[l]_j
        for (unsigned j = 0; j < b.n(); j++)
        {
          b[j] -= learning_rate * delta_current[j];
        }

        // Update weights: w[l]_jk ← w[l]_jk - η * δ[l]_j * a[l-1]_k
        for (unsigned j = 0; j < W_ref.n(); j++)
        {
          for (unsigned k = 0; k < W_ref.m(); k++)
          {
            W_ref(j, k) -= learning_rate * delta_current[j] * a_prev[k];
          }
        }

        // Swap delta for next iteration
        if (l > 0)
        {
          delta_current = delta_next;
        }
      }

      iter++;

      // Check convergence every 1000 iterations
      if (iter % 1000 == 0)
      {
        current_cost = cost_for_training_data(training_data);

        DEBUG_PRINT("Iteration " << iter << ": cost = " << current_cost);

        if (!convergence_history_file_name.empty())
        {
          history_file << iter << " " << current_cost << std::endl;
        }

        if (current_cost < tol_training)
        {
          std::cout << "Training converged at iteration " << iter
                    << " with cost " << current_cost << std::endl;
          if (history_file.is_open())
          {
            history_file.close();
          }
          return;
        }
      }
    }

    // Training did not converge
    std::cout << "Training did not converge after " << max_iter
              << " iterations. Final cost: " << current_cost << std::endl;

    if (history_file.is_open())
    {
      history_file.close();
    }
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

  /// Public wrapper for finite-difference gradient computation (for testing)
  void test_compute_gradients_finite_difference(
    const DoubleVector& input,
    const DoubleVector& target,
    std::vector<DoubleMatrix>& grad_W,
    std::vector<DoubleVector>& grad_b,
    const double& epsilon = 1.0e-7)
  {
    compute_gradients_finite_difference(input, target, grad_W, grad_b, epsilon);
  }

  /// Public wrapper for back-propagation gradient computation (for testing)
  void test_compute_gradients_backprop(
    const DoubleVector& input,
    const DoubleVector& target,
    std::vector<DoubleMatrix>& grad_W,
    std::vector<DoubleVector>& grad_b)
  {
    compute_gradients_backprop(input, target, grad_W, grad_b);
  }
};

#endif // PROJECT2_A_H
