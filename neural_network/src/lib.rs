pub mod activation;
pub mod initializer;
pub mod optimizer;

use linear_algebra::{matrix::Matrix, vectors::Vector};
use activation::ActivationType;
use initializer::Initializer;

/// A single layer in the neural network
#[derive(Debug, Clone)]
pub struct Layer {
    /// Weight matrix (n_outputs × n_inputs)
    pub weights: Matrix<f64>,

    /// Bias vector (n_outputs)
    pub biases: Vector<f64>,

    /// Activation function for this layer
    pub activation: ActivationType,

    /// Cached neuron activations from forward pass (for visualization)
    pub activations: Vector<f64>,

    /// Cached pre-activation values (z = Wx + b) for backprop
    pub z_values: Vector<f64>,
}

impl Layer {
    /// Create a new layer with specified dimensions
    pub fn new(
        n_inputs: usize,
        n_outputs: usize,
        activation: ActivationType,
        initializer: Initializer,
    ) -> Self {
        let weights = initializer.initialize_matrix(n_outputs, n_inputs);
        let biases = initializer.initialize_vector(n_outputs);
        let activations = Vector { data: vec![0.0; n_outputs] };
        let z_values = Vector { data: vec![0.0; n_outputs] };

        Layer {
            weights,
            biases,
            activation,
            activations,
            z_values,
        }
    }

    /// Forward pass through this layer
    ///
    /// Computes: a = activation(Wx + b)
    pub fn forward(&mut self, input: &Vector<f64>) -> Vector<f64> {
        // z = Wx + b
        self.z_values = (self.weights.clone() * input.clone()) + self.biases.clone();

        // a = activation(z)
        self.activations = Vector {
            data: self.activation.activate_vector(&self.z_values.data),
        };

        self.activations.clone()
    }
}

/// Training history for visualization
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    /// Loss value at each epoch
    pub losses: Vec<f64>,

    /// Accuracy at each epoch (for classification)
    pub accuracies: Vec<f64>,

    /// Weight snapshots for animation (every N epochs)
    pub weight_snapshots: Vec<Vec<Matrix<f64>>>,

    /// Activation snapshots for visualization
    pub activation_snapshots: Vec<Vec<Vector<f64>>>,
}

impl TrainingHistory {
    pub fn new() -> Self {
        TrainingHistory {
            losses: Vec::new(),
            accuracies: Vec::new(),
            weight_snapshots: Vec::new(),
            activation_snapshots: Vec::new(),
        }
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-layer neural network
#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    /// Network layers (input to output)
    pub layers: Vec<Layer>,

    /// Learning rate for gradient descent
    pub learning_rate: f64,

    /// Training history for visualization
    pub history: TrainingHistory,
}

impl NeuralNetwork {
    /// Create a new neural network
    ///
    /// # Arguments
    /// * `layer_sizes` - Sizes of each layer [input, hidden1, ..., output]
    /// * `activations` - Activation function for each layer (length = layer_sizes.len() - 1)
    /// * `learning_rate` - Learning rate for gradient descent
    ///
    /// # Example
    /// ```
    /// use neural_network::{NeuralNetwork, activation::ActivationType};
    ///
    /// // Create 2-4-3-2 network (2 inputs, 2 hidden layers, 2 outputs)
    /// let nn = NeuralNetwork::new(
    ///     &[2, 4, 3, 2],
    ///     &[ActivationType::ReLU, ActivationType::ReLU, ActivationType::Sigmoid],
    ///     0.01
    /// );
    /// ```
    pub fn new(
        layer_sizes: &[usize],
        activations: &[ActivationType],
        learning_rate: f64,
    ) -> Self {
        assert!(layer_sizes.len() >= 2, "Need at least input and output layer");
        assert_eq!(layer_sizes.len() - 1, activations.len(),
                   "Need one activation per layer (excluding input)");

        let mut layers = Vec::new();

        for i in 0..activations.len() {
            let n_inputs = layer_sizes[i];
            let n_outputs = layer_sizes[i + 1];

            // Use He init for ReLU, Xavier for others
            let initializer = match activations[i] {
                ActivationType::ReLU => Initializer::He,
                _ => Initializer::Xavier,
            };

            layers.push(Layer::new(n_inputs, n_outputs, activations[i], initializer));
        }

        NeuralNetwork {
            layers,
            learning_rate,
            history: TrainingHistory::new(),
        }
    }

    /// Forward propagation through the entire network
    ///
    /// # Arguments
    /// * `input` - Input vector
    ///
    /// # Returns
    /// Output from the final layer
    pub fn forward(&mut self, input: &Vector<f64>) -> Vector<f64> {
        let mut activation = input.clone();

        for layer in &mut self.layers {
            activation = layer.forward(&activation);
        }

        activation
    }

    /// Backward propagation and weight update
    ///
    /// # Arguments
    /// * `input` - Input vector
    /// * `target` - Target output vector
    ///
    /// # Returns
    /// Loss value for this sample
    pub fn backward(&mut self, input: &Vector<f64>, target: &Vector<f64>) -> f64 {
        // Forward pass
        let output = self.forward(input);

        // Compute loss (Mean Squared Error)
        let loss = Self::mse_loss(&output, target);

        // Compute output layer error
        // δ^L = (a^L - y) ⊙ σ'(z^L)
        let last_idx = self.layers.len() - 1;
        let output_error = Self::compute_output_error(
            &output,
            target,
            &self.layers[last_idx].z_values,
            self.layers[last_idx].activation,
        );

        // Store errors for each layer
        let mut errors = vec![Vector { data: vec![] }; self.layers.len()];
        errors[last_idx] = output_error;

        // Backpropagate errors through hidden layers
        for l in (0..last_idx).rev() {
            // δ^l = (W^{l+1})^T δ^{l+1} ⊙ σ'(z^l)
            let next_weights = &self.layers[l + 1].weights;
            let next_error = &errors[l + 1];

            let error = Self::backpropagate_error(
                next_weights,
                next_error,
                &self.layers[l].z_values,
                self.layers[l].activation,
            );

            errors[l] = error;
        }

        // Update weights and biases
        let mut current_activation = input.clone();

        for (l, layer) in self.layers.iter_mut().enumerate() {
            // ∂L/∂W = δ * a^{l-1}^T
            let weight_gradient = Self::compute_weight_gradient(&errors[l], &current_activation);

            // ∂L/∂b = δ
            let bias_gradient = errors[l].clone();

            // W := W - α * ∂L/∂W
            layer.weights = layer.weights.clone() - weight_gradient * self.learning_rate;

            // b := b - α * ∂L/∂b
            layer.biases = layer.biases.clone() - bias_gradient * self.learning_rate;

            // Update current activation for next layer
            current_activation = layer.activations.clone();
        }

        loss
    }

    /// Train the network on a batch of data for one epoch
    ///
    /// # Arguments
    /// * `X` - Input matrix (n_samples × n_features)
    /// * `y` - Target matrix (n_samples × n_outputs)
    ///
    /// # Returns
    /// Average loss for this epoch
    pub fn train_epoch(&mut self, X: &Matrix<f64>, y: &Matrix<f64>) -> f64 {
        assert_eq!(X.rows, y.rows, "X and y must have same number of samples");

        let mut total_loss = 0.0;
        let n_samples = X.rows;

        for i in 0..n_samples {
            let input = X.row(i).unwrap();
            let target = y.row(i).unwrap();

            let loss = self.backward(&input, &target);
            total_loss += loss;
        }

        total_loss / n_samples as f64
    }

    /// Train the network for multiple epochs
    ///
    /// # Arguments
    /// * `X` - Input matrix (n_samples × n_features)
    /// * `y` - Target matrix (n_samples × n_outputs)
    /// * `epochs` - Number of training epochs
    /// * `snapshot_interval` - Save snapshots every N epochs (0 = no snapshots)
    pub fn fit(&mut self, X: &Matrix<f64>, y: &Matrix<f64>, epochs: usize, snapshot_interval: usize) {
        self.history = TrainingHistory::new();

        for epoch in 0..epochs {
            let loss = self.train_epoch(X, y);
            self.history.losses.push(loss);

            // Compute accuracy (for classification)
            let accuracy = self.compute_accuracy(X, y);
            self.history.accuracies.push(accuracy);

            // Save snapshots for animation
            if snapshot_interval > 0 && epoch % snapshot_interval == 0 {
                let weight_snapshot: Vec<Matrix<f64>> = self.layers.iter()
                    .map(|layer| layer.weights.clone())
                    .collect();
                self.history.weight_snapshots.push(weight_snapshot);

                let activation_snapshot: Vec<Vector<f64>> = self.layers.iter()
                    .map(|layer| layer.activations.clone())
                    .collect();
                self.history.activation_snapshots.push(activation_snapshot);
            }
        }
    }

    /// Make predictions on input data
    ///
    /// # Arguments
    /// * `X` - Input matrix (n_samples × n_features)
    ///
    /// # Returns
    /// Predictions matrix (n_samples × n_outputs)
    pub fn predict(&mut self, X: &Matrix<f64>) -> Matrix<f64> {
        let mut predictions = Vec::new();

        for i in 0..X.rows {
            let input = X.row(i).unwrap();
            let output = self.forward(&input);
            predictions.extend(output.data);
        }

        Matrix::from_vec(predictions, X.rows, self.layers.last().unwrap().weights.rows).unwrap()
    }

    /// Mean Squared Error loss
    fn mse_loss(predictions: &Vector<f64>, targets: &Vector<f64>) -> f64 {
        assert_eq!(predictions.data.len(), targets.data.len());

        let sum_squared_error: f64 = predictions.data.iter()
            .zip(targets.data.iter())
            .map(|(pred, target)| (pred - target).powi(2))
            .sum();

        sum_squared_error / predictions.data.len() as f64
    }

    /// Compute output layer error: δ^L = (a^L - y) ⊙ σ'(z^L)
    fn compute_output_error(
        output: &Vector<f64>,
        target: &Vector<f64>,
        z_values: &Vector<f64>,
        activation: ActivationType,
    ) -> Vector<f64> {
        let derivatives = activation.derivative_vector(&z_values.data);

        let error_data: Vec<f64> = output.data.iter()
            .zip(target.data.iter())
            .zip(derivatives.iter())
            .map(|((out, targ), deriv)| (out - targ) * deriv)
            .collect();

        Vector { data: error_data }
    }

    /// Backpropagate error: δ^l = (W^{l+1})^T δ^{l+1} ⊙ σ'(z^l)
    fn backpropagate_error(
        next_weights: &Matrix<f64>,
        next_error: &Vector<f64>,
        z_values: &Vector<f64>,
        activation: ActivationType,
    ) -> Vector<f64> {
        // (W^{l+1})^T δ^{l+1}
        let transposed = next_weights.transpose();
        let weighted_error = transposed * next_error.clone();

        // σ'(z^l)
        let derivatives = activation.derivative_vector(&z_values.data);

        // Element-wise multiplication
        let error_data: Vec<f64> = weighted_error.data.iter()
            .zip(derivatives.iter())
            .map(|(err, deriv)| err * deriv)
            .collect();

        Vector { data: error_data }
    }

    /// Compute weight gradient: ∂L/∂W = δ * a^{l-1}^T
    fn compute_weight_gradient(error: &Vector<f64>, prev_activation: &Vector<f64>) -> Matrix<f64> {
        let n_outputs = error.data.len();
        let n_inputs = prev_activation.data.len();

        let mut gradient_data = Vec::with_capacity(n_outputs * n_inputs);

        for &err in &error.data {
            for &act in &prev_activation.data {
                gradient_data.push(err * act);
            }
        }

        Matrix::from_vec(gradient_data, n_outputs, n_inputs).unwrap()
    }

    /// Compute classification accuracy
    fn compute_accuracy(&mut self, X: &Matrix<f64>, y: &Matrix<f64>) -> f64 {
        let predictions = self.predict(X);
        let mut correct = 0;

        for i in 0..X.rows {
            let pred_row = predictions.row(i).unwrap();
            let target_row = y.row(i).unwrap();

            // Find argmax for both
            let pred_class = Self::argmax(&pred_row.data);
            let target_class = Self::argmax(&target_row.data);

            if pred_class == target_class {
                correct += 1;
            }
        }

        (correct as f64 / X.rows as f64) * 100.0
    }

    /// Find index of maximum value
    fn argmax(values: &[f64]) -> usize {
        values.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_layer_creation() {
        let layer = Layer::new(3, 2, ActivationType::ReLU, Initializer::Xavier);

        assert_eq!(layer.weights.rows, 2);
        assert_eq!(layer.weights.cols, 3);
        assert_eq!(layer.biases.data.len(), 2);
    }

    #[test]
    fn test_layer_forward() {
        let mut layer = Layer::new(3, 2, ActivationType::ReLU, Initializer::Zeros);

        // With zero weights, ReLU should output zeros
        let input = Vector { data: vec![1.0, 2.0, 3.0] };
        let output = layer.forward(&input);

        assert_eq!(output.data.len(), 2);
        assert_eq!(output.data[0], 0.0);
        assert_eq!(output.data[1], 0.0);
    }

    #[test]
    fn test_network_creation() {
        let nn = NeuralNetwork::new(
            &[2, 4, 3, 2],
            &[ActivationType::ReLU, ActivationType::ReLU, ActivationType::Sigmoid],
            0.01,
        );

        assert_eq!(nn.layers.len(), 3);
        assert_eq!(nn.layers[0].weights.cols, 2);  // Input layer
        assert_eq!(nn.layers[2].weights.rows, 2);  // Output layer
    }

    #[test]
    fn test_forward_pass() {
        let mut nn = NeuralNetwork::new(
            &[2, 3, 1],
            &[ActivationType::ReLU, ActivationType::Sigmoid],
            0.01,
        );

        let input = Vector { data: vec![1.0, 0.5] };
        let output = nn.forward(&input);

        assert_eq!(output.data.len(), 1);
        // Output should be in (0, 1) due to sigmoid
        assert!(output.data[0] > 0.0 && output.data[0] < 1.0);
    }

    #[test]
    fn test_xor_problem() {
        // XOR is the classic non-linearly separable problem
        let X = Matrix::from_vec(
            vec![
                0.0, 0.0,
                0.0, 1.0,
                1.0, 0.0,
                1.0, 1.0,
            ],
            4,
            2,
        ).unwrap();

        let y = Matrix::from_vec(
            vec![
                0.0,
                1.0,
                1.0,
                0.0,
            ],
            4,
            1,
        ).unwrap();

        let mut nn = NeuralNetwork::new(
            &[2, 4, 1],
            &[ActivationType::Tanh, ActivationType::Sigmoid],
            0.5,
        );

        // Train for 1000 epochs
        nn.fit(&X, &y, 1000, 0);

        // Check that loss decreased
        assert!(nn.history.losses.last().unwrap() < nn.history.losses.first().unwrap());

        // Check predictions
        let predictions = nn.predict(&X);

        // Should be close to XOR outputs
        assert!(predictions[(0, 0)] < 0.3);  // 0 XOR 0 = 0
        assert!(predictions[(1, 0)] > 0.7);  // 0 XOR 1 = 1
        assert!(predictions[(2, 0)] > 0.7);  // 1 XOR 0 = 1
        assert!(predictions[(3, 0)] < 0.3);  // 1 XOR 1 = 0
    }

    #[test]
    fn test_mse_loss() {
        let predictions = Vector { data: vec![0.9, 0.1, 0.5] };
        let targets = Vector { data: vec![1.0, 0.0, 0.5] };

        let loss = NeuralNetwork::mse_loss(&predictions, &targets);

        // (0.1^2 + 0.1^2 + 0^2) / 3 = 0.02 / 3
        assert_relative_eq!(loss, 0.00666666, epsilon = 1e-5);
    }

    #[test]
    fn test_argmax() {
        let values = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        assert_eq!(NeuralNetwork::argmax(&values), 3);

        let values2 = vec![0.9, 0.1];
        assert_eq!(NeuralNetwork::argmax(&values2), 0);
    }
}
