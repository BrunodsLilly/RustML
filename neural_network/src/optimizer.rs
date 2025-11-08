//! Gradient Descent Optimization Algorithms
//!
//! This module implements various gradient descent optimization algorithms including:
//! - SGD (Stochastic Gradient Descent)
//! - Momentum
//! - RMSprop
//! - Adam (Adaptive Moment Estimation)
//!
//! # Examples
//!
//! ```
//! use neural_network::optimizer::{Optimizer, OptimizerType};
//!
//! // Create Adam optimizer with default hyperparameters
//! let optimizer = Optimizer::adam(0.001, 0.9, 0.999, 1e-8);
//!
//! // Or use builder pattern
//! let optimizer = Optimizer::new(OptimizerType::SGD, 0.001);
//! ```

use linear_algebra::{matrix::Matrix, vectors::Vector};

/// Optimizer algorithm types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizerType {
    /// Vanilla stochastic gradient descent: θ = θ - α∇L
    SGD,
    /// Momentum-based gradient descent: v = β₁·v + ∇L; θ = θ - α·v
    Momentum,
    /// RMSprop with adaptive per-parameter learning rates
    RMSprop,
    /// Adam: Combines Momentum + RMSprop with bias correction
    Adam,
}

/// Gradient descent optimizer with support for multiple algorithms
#[derive(Debug, Clone, PartialEq)]
pub struct Optimizer {
    /// Type of optimization algorithm
    optimizer_type: OptimizerType,
    /// Learning rate (step size)
    learning_rate: f64,

    // Momentum state (used by Momentum and Adam)
    /// Velocity for weights (first moment)
    velocity_weights: Vec<Matrix<f64>>,
    /// Velocity for biases (first moment)
    velocity_bias: Vec<Vector<f64>>,
    /// Momentum decay coefficient (β₁)
    beta1: f64,

    // RMSprop/Adam state (used by RMSprop and Adam)
    /// Squared gradients for weights (second moment)
    squared_gradients_weights: Vec<Matrix<f64>>,
    /// Squared gradients for biases (second moment)
    squared_gradients_bias: Vec<Vector<f64>>,
    /// RMS decay coefficient (β₂)
    beta2: f64,
    /// Small constant for numerical stability
    epsilon: f64,

    // Adam-specific state
    /// Timestep counter for bias correction
    timestep: usize,
}

impl Optimizer {
    /// Create a new optimizer with the specified type and learning rate
    pub fn new(optimizer_type: OptimizerType, learning_rate: f64) -> Self {
        match optimizer_type {
            OptimizerType::SGD => Self::sgd(learning_rate),
            OptimizerType::Momentum => Self::momentum(learning_rate, 0.9),
            OptimizerType::RMSprop => Self::rmsprop(learning_rate, 0.999, 1e-8),
            OptimizerType::Adam => Self::adam(learning_rate, 0.9, 0.999, 1e-8),
        }
    }

    /// Create an SGD optimizer
    ///
    /// # Algorithm
    /// ```text
    /// θ = θ - α∇L(θ)
    /// ```
    ///
    /// # Arguments
    /// * `learning_rate` - Step size (α)
    pub fn sgd(learning_rate: f64) -> Self {
        Optimizer {
            optimizer_type: OptimizerType::SGD,
            learning_rate,
            velocity_weights: Vec::new(),
            velocity_bias: Vec::new(),
            beta1: 0.0,
            squared_gradients_weights: Vec::new(),
            squared_gradients_bias: Vec::new(),
            beta2: 0.0,
            epsilon: 0.0,
            timestep: 0,
        }
    }

    /// Create a Momentum optimizer
    ///
    /// # Algorithm
    /// ```text
    /// v = β₁·v + ∇L(θ)
    /// θ = θ - α·v
    /// ```
    ///
    /// # Arguments
    /// * `learning_rate` - Step size (α)
    /// * `beta1` - Momentum decay coefficient (typical: 0.9)
    pub fn momentum(learning_rate: f64, beta1: f64) -> Self {
        Optimizer {
            optimizer_type: OptimizerType::Momentum,
            learning_rate,
            velocity_weights: Vec::new(),
            velocity_bias: Vec::new(),
            beta1,
            squared_gradients_weights: Vec::new(),
            squared_gradients_bias: Vec::new(),
            beta2: 0.0,
            epsilon: 0.0,
            timestep: 0,
        }
    }

    /// Create an RMSprop optimizer
    ///
    /// # Algorithm
    /// ```text
    /// s = β₂·s + (1-β₂)·(∇L(θ))²
    /// θ = θ - α·∇L(θ)/√(s+ε)
    /// ```
    ///
    /// # Arguments
    /// * `learning_rate` - Step size (α)
    /// * `beta2` - RMS decay coefficient (typical: 0.999)
    /// * `epsilon` - Numerical stability constant (typical: 1e-8)
    pub fn rmsprop(learning_rate: f64, beta2: f64, epsilon: f64) -> Self {
        Optimizer {
            optimizer_type: OptimizerType::RMSprop,
            learning_rate,
            velocity_weights: Vec::new(),
            velocity_bias: Vec::new(),
            beta1: 0.0,
            squared_gradients_weights: Vec::new(),
            squared_gradients_bias: Vec::new(),
            beta2,
            epsilon,
            timestep: 0,
        }
    }

    /// Create an Adam optimizer
    ///
    /// # Algorithm
    /// ```text
    /// m = β₁·m + (1-β₁)·∇L(θ)
    /// v = β₂·v + (1-β₂)·(∇L(θ))²
    /// m̂ = m / (1 - β₁ᵗ)
    /// v̂ = v / (1 - β₂ᵗ)
    /// θ = θ - α·m̂/√(v̂ + ε)
    /// ```
    ///
    /// # Arguments
    /// * `learning_rate` - Step size (α)
    /// * `beta1` - First moment decay (typical: 0.9)
    /// * `beta2` - Second moment decay (typical: 0.999)
    /// * `epsilon` - Numerical stability constant (typical: 1e-8)
    pub fn adam(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Optimizer {
            optimizer_type: OptimizerType::Adam,
            learning_rate,
            velocity_weights: Vec::new(),
            velocity_bias: Vec::new(),
            beta1,
            squared_gradients_weights: Vec::new(),
            squared_gradients_bias: Vec::new(),
            beta2,
            epsilon,
            timestep: 0,
        }
    }

    /// Initialize optimizer state for a network with the given layer shapes
    fn initialize_state(&mut self, layer_shapes: &[(usize, usize)]) {
        match self.optimizer_type {
            OptimizerType::SGD => {
                // SGD has no state
            }
            OptimizerType::Momentum => {
                // Initialize velocity
                self.velocity_weights = layer_shapes
                    .iter()
                    .map(|&(rows, cols)| Matrix::zeros(rows, cols))
                    .collect();
                self.velocity_bias = layer_shapes
                    .iter()
                    .map(|&(rows, _)| Vector { data: vec![0.0; rows] })
                    .collect();
            }
            OptimizerType::RMSprop => {
                // Initialize squared gradients
                self.squared_gradients_weights = layer_shapes
                    .iter()
                    .map(|&(rows, cols)| Matrix::zeros(rows, cols))
                    .collect();
                self.squared_gradients_bias = layer_shapes
                    .iter()
                    .map(|&(rows, _)| Vector { data: vec![0.0; rows] })
                    .collect();
            }
            OptimizerType::Adam => {
                // Initialize both velocity and squared gradients
                self.velocity_weights = layer_shapes
                    .iter()
                    .map(|&(rows, cols)| Matrix::zeros(rows, cols))
                    .collect();
                self.velocity_bias = layer_shapes
                    .iter()
                    .map(|&(rows, _)| Vector { data: vec![0.0; rows] })
                    .collect();
                self.squared_gradients_weights = layer_shapes
                    .iter()
                    .map(|&(rows, cols)| Matrix::zeros(rows, cols))
                    .collect();
                self.squared_gradients_bias = layer_shapes
                    .iter()
                    .map(|&(rows, _)| Vector { data: vec![0.0; rows] })
                    .collect();
            }
        }
    }

    /// Update weights using the optimizer's algorithm
    ///
    /// # Arguments
    /// * `layer_idx` - Index of the layer being updated
    /// * `gradient` - Gradient of the loss with respect to the weights
    /// * `weights` - Current weights (will be updated in-place)
    /// * `layer_shapes` - Shapes of all layers (for lazy initialization)
    pub fn update_weights(
        &mut self,
        layer_idx: usize,
        gradient: &Matrix<f64>,
        weights: &mut Matrix<f64>,
        layer_shapes: &[(usize, usize)],
    ) {
        // Lazy initialization on first use
        if self.requires_state() && self.velocity_weights.is_empty() && self.squared_gradients_weights.is_empty() {
            self.initialize_state(layer_shapes);
        }

        match self.optimizer_type {
            OptimizerType::SGD => {
                // θ = θ - α∇L
                for i in 0..weights.rows {
                    for j in 0..weights.cols {
                        weights[(i, j)] -= self.learning_rate * gradient[(i, j)];
                    }
                }
            }
            OptimizerType::Momentum => {
                let v = &mut self.velocity_weights[layer_idx];

                // v = β₁·v + ∇L
                // θ = θ - α·v
                for i in 0..weights.rows {
                    for j in 0..weights.cols {
                        v[(i, j)] = self.beta1 * v[(i, j)] + gradient[(i, j)];
                        weights[(i, j)] -= self.learning_rate * v[(i, j)];
                    }
                }
            }
            OptimizerType::RMSprop => {
                let s = &mut self.squared_gradients_weights[layer_idx];

                // s = β₂·s + (1-β₂)·(∇L)²
                // θ = θ - α·∇L/√(s+ε)
                for i in 0..weights.rows {
                    for j in 0..weights.cols {
                        let grad = gradient[(i, j)];
                        s[(i, j)] = self.beta2 * s[(i, j)] + (1.0 - self.beta2) * grad * grad;
                        weights[(i, j)] -= self.learning_rate * grad / (s[(i, j)] + self.epsilon).sqrt();
                    }
                }
            }
            OptimizerType::Adam => {
                self.timestep += 1;
                let t = self.timestep as f64;

                let m = &mut self.velocity_weights[layer_idx];
                let v = &mut self.squared_gradients_weights[layer_idx];

                // m = β₁·m + (1-β₁)·∇L
                // v = β₂·v + (1-β₂)·(∇L)²
                // m̂ = m / (1 - β₁ᵗ)
                // v̂ = v / (1 - β₂ᵗ)
                // θ = θ - α·m̂/√(v̂ + ε)
                for i in 0..weights.rows {
                    for j in 0..weights.cols {
                        let grad = gradient[(i, j)];

                        m[(i, j)] = self.beta1 * m[(i, j)] + (1.0 - self.beta1) * grad;
                        v[(i, j)] = self.beta2 * v[(i, j)] + (1.0 - self.beta2) * grad * grad;

                        // Bias correction
                        let m_hat = m[(i, j)] / (1.0 - self.beta1.powf(t));
                        let v_hat = v[(i, j)] / (1.0 - self.beta2.powf(t));

                        weights[(i, j)] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
                    }
                }
            }
        }
    }

    /// Update biases using the optimizer's algorithm
    ///
    /// # Arguments
    /// * `layer_idx` - Index of the layer being updated
    /// * `gradient` - Gradient of the loss with respect to the biases
    /// * `bias` - Current biases (will be updated in-place)
    /// * `layer_shapes` - Shapes of all layers (for lazy initialization)
    pub fn update_bias(
        &mut self,
        layer_idx: usize,
        gradient: &Vector<f64>,
        bias: &mut Vector<f64>,
        layer_shapes: &[(usize, usize)],
    ) {
        // Lazy initialization on first use
        if self.requires_state() && self.velocity_bias.is_empty() && self.squared_gradients_bias.is_empty() {
            self.initialize_state(layer_shapes);
        }

        match self.optimizer_type {
            OptimizerType::SGD => {
                // b = b - α∇L
                for i in 0..bias.data.len() {
                    bias.data[i] -= self.learning_rate * gradient.data[i];
                }
            }
            OptimizerType::Momentum => {
                let v = &mut self.velocity_bias[layer_idx];

                // v = β₁·v + ∇L
                // b = b - α·v
                for i in 0..bias.data.len() {
                    v.data[i] = self.beta1 * v.data[i] + gradient.data[i];
                    bias.data[i] -= self.learning_rate * v.data[i];
                }
            }
            OptimizerType::RMSprop => {
                let s = &mut self.squared_gradients_bias[layer_idx];

                // s = β₂·s + (1-β₂)·(∇L)²
                // b = b - α·∇L/√(s+ε)
                for i in 0..bias.data.len() {
                    let grad = gradient.data[i];
                    s.data[i] = self.beta2 * s.data[i] + (1.0 - self.beta2) * grad * grad;
                    bias.data[i] -= self.learning_rate * grad / (s.data[i] + self.epsilon).sqrt();
                }
            }
            OptimizerType::Adam => {
                // Note: timestep is incremented once per iteration (not per layer)
                // So we don't increment it here, only in update_weights

                let t = self.timestep.max(1) as f64; // Prevent division by zero

                let m = &mut self.velocity_bias[layer_idx];
                let v = &mut self.squared_gradients_bias[layer_idx];

                for i in 0..bias.data.len() {
                    let grad = gradient.data[i];

                    m.data[i] = self.beta1 * m.data[i] + (1.0 - self.beta1) * grad;
                    v.data[i] = self.beta2 * v.data[i] + (1.0 - self.beta2) * grad * grad;

                    // Bias correction
                    let m_hat = m.data[i] / (1.0 - self.beta1.powf(t));
                    let v_hat = v.data[i] / (1.0 - self.beta2.powf(t));

                    bias.data[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
                }
            }
        }
    }

    /// Reset optimizer state (useful when restarting training)
    pub fn reset(&mut self) {
        self.velocity_weights.clear();
        self.velocity_bias.clear();
        self.squared_gradients_weights.clear();
        self.squared_gradients_bias.clear();
        self.timestep = 0;
    }

    /// Check if this optimizer requires state storage
    fn requires_state(&self) -> bool {
        !matches!(self.optimizer_type, OptimizerType::SGD)
    }

    /// Get the optimizer type
    pub fn optimizer_type(&self) -> OptimizerType {
        self.optimizer_type
    }

    /// Get the learning rate
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Set the learning rate
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    /// Get beta1 (momentum coefficient)
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Get beta2 (RMS coefficient)
    pub fn beta2(&self) -> f64 {
        self.beta2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_creation() {
        let opt = Optimizer::sgd(0.01);
        assert_eq!(opt.optimizer_type(), OptimizerType::SGD);
        assert_eq!(opt.learning_rate(), 0.01);
    }

    #[test]
    fn test_momentum_creation() {
        let opt = Optimizer::momentum(0.01, 0.9);
        assert_eq!(opt.optimizer_type(), OptimizerType::Momentum);
        assert_eq!(opt.learning_rate(), 0.01);
        assert_eq!(opt.beta1(), 0.9);
    }

    #[test]
    fn test_rmsprop_creation() {
        let opt = Optimizer::rmsprop(0.001, 0.999, 1e-8);
        assert_eq!(opt.optimizer_type(), OptimizerType::RMSprop);
        assert_eq!(opt.learning_rate(), 0.001);
        assert_eq!(opt.beta2(), 0.999);
    }

    #[test]
    fn test_adam_creation() {
        let opt = Optimizer::adam(0.001, 0.9, 0.999, 1e-8);
        assert_eq!(opt.optimizer_type(), OptimizerType::Adam);
        assert_eq!(opt.learning_rate(), 0.001);
        assert_eq!(opt.beta1(), 0.9);
        assert_eq!(opt.beta2(), 0.999);
    }

    #[test]
    fn test_sgd_simple_update() {
        let mut opt = Optimizer::sgd(0.1);
        let mut weights = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let gradient = Matrix::from_vec(vec![0.1, 0.2, 0.3, 0.4], 2, 2).unwrap();
        let layer_shapes = vec![(2, 2)];

        opt.update_weights(0, &gradient, &mut weights, &layer_shapes);

        // θ_new = θ_old - α * ∇L = [1,2,3,4] - 0.1 * [0.1,0.2,0.3,0.4]
        assert!((weights[(0, 0)] - 0.99).abs() < 1e-10);
        assert!((weights[(0, 1)] - 1.98).abs() < 1e-10);
        assert!((weights[(1, 0)] - 2.97).abs() < 1e-10);
        assert!((weights[(1, 1)] - 3.96).abs() < 1e-10);
    }
}
