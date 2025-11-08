/// Activation function types supported by the neural network
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationType {
    /// Sigmoid: σ(x) = 1 / (1 + e^(-x))
    /// Range: (0, 1)
    /// Use case: Binary classification output layer
    Sigmoid,

    /// ReLU: f(x) = max(0, x)
    /// Range: [0, ∞)
    /// Use case: Hidden layers (fast, prevents vanishing gradients)
    ReLU,

    /// Tanh: f(x) = tanh(x)
    /// Range: (-1, 1)
    /// Use case: Hidden layers (zero-centered)
    Tanh,

    /// Linear: f(x) = x
    /// Range: (-∞, ∞)
    /// Use case: Regression output layer
    Linear,
}

impl ActivationType {
    /// Apply the activation function to a single value
    pub fn activate(&self, z: f64) -> f64 {
        match self {
            ActivationType::Sigmoid => 1.0 / (1.0 + (-z).exp()),
            ActivationType::ReLU => z.max(0.0),
            ActivationType::Tanh => z.tanh(),
            ActivationType::Linear => z,
        }
    }

    /// Compute the derivative of the activation function
    ///
    /// For backpropagation, we need ∂a/∂z where:
    /// - a = activation(z)
    /// - z = pre-activation value
    pub fn derivative(&self, z: f64) -> f64 {
        match self {
            ActivationType::Sigmoid => {
                let a = self.activate(z);
                a * (1.0 - a)
            }
            ActivationType::ReLU => {
                if z > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationType::Tanh => {
                let a = self.activate(z);
                1.0 - a * a
            }
            ActivationType::Linear => 1.0,
        }
    }

    /// Apply activation function to a vector of values
    pub fn activate_vector(&self, z: &[f64]) -> Vec<f64> {
        z.iter().map(|&val| self.activate(val)).collect()
    }

    /// Compute derivatives for a vector of values
    pub fn derivative_vector(&self, z: &[f64]) -> Vec<f64> {
        z.iter().map(|&val| self.derivative(val)).collect()
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            ActivationType::Sigmoid => "Sigmoid",
            ActivationType::ReLU => "ReLU",
            ActivationType::Tanh => "Tanh",
            ActivationType::Linear => "Linear",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sigmoid_activation() {
        let sigmoid = ActivationType::Sigmoid;

        // Test at zero
        assert_relative_eq!(sigmoid.activate(0.0), 0.5, epsilon = 1e-10);

        // Test positive values
        assert!(sigmoid.activate(5.0) > 0.9);
        assert!(sigmoid.activate(10.0) > 0.99);

        // Test negative values
        assert!(sigmoid.activate(-5.0) < 0.1);

        // Test range (0, 1)
        for x in -10..=10 {
            let val = sigmoid.activate(x as f64);
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_sigmoid_derivative() {
        let sigmoid = ActivationType::Sigmoid;

        // At x=0, sigmoid(0) = 0.5, derivative = 0.25
        assert_relative_eq!(sigmoid.derivative(0.0), 0.25, epsilon = 1e-10);

        // Derivative should be positive
        assert!(sigmoid.derivative(1.0) > 0.0);
        assert!(sigmoid.derivative(-1.0) > 0.0);

        // Derivative approaches 0 at extremes
        assert!(sigmoid.derivative(10.0) < 0.01);
        assert!(sigmoid.derivative(-10.0) < 0.01);
    }

    #[test]
    fn test_relu_activation() {
        let relu = ActivationType::ReLU;

        // Positive values pass through
        assert_eq!(relu.activate(5.0), 5.0);
        assert_eq!(relu.activate(100.0), 100.0);

        // Negative values become zero
        assert_eq!(relu.activate(-5.0), 0.0);
        assert_eq!(relu.activate(-100.0), 0.0);

        // Zero stays zero
        assert_eq!(relu.activate(0.0), 0.0);
    }

    #[test]
    fn test_relu_derivative() {
        let relu = ActivationType::ReLU;

        // Derivative is 1 for positive
        assert_eq!(relu.derivative(5.0), 1.0);
        assert_eq!(relu.derivative(100.0), 1.0);

        // Derivative is 0 for negative
        assert_eq!(relu.derivative(-5.0), 0.0);
        assert_eq!(relu.derivative(-100.0), 0.0);

        // Derivative at zero is 0 (by convention)
        assert_eq!(relu.derivative(0.0), 0.0);
    }

    #[test]
    fn test_tanh_activation() {
        let tanh = ActivationType::Tanh;

        // Test at zero
        assert_relative_eq!(tanh.activate(0.0), 0.0, epsilon = 1e-10);

        // Test range (-1, 1)
        for x in -10..=10 {
            let val = tanh.activate(x as f64);
            assert!(val > -1.0 && val < 1.0);
        }

        // Test saturation
        assert!(tanh.activate(5.0) > 0.99);
        assert!(tanh.activate(-5.0) < -0.99);
    }

    #[test]
    fn test_tanh_derivative() {
        let tanh = ActivationType::Tanh;

        // At x=0, tanh(0) = 0, derivative = 1
        assert_relative_eq!(tanh.derivative(0.0), 1.0, epsilon = 1e-10);

        // Derivative should be positive
        assert!(tanh.derivative(1.0) > 0.0);
        assert!(tanh.derivative(-1.0) > 0.0);

        // Derivative approaches 0 at extremes
        assert!(tanh.derivative(5.0) < 0.01);
        assert!(tanh.derivative(-5.0) < 0.01);
    }

    #[test]
    fn test_linear_activation() {
        let linear = ActivationType::Linear;

        // Identity function
        assert_eq!(linear.activate(5.0), 5.0);
        assert_eq!(linear.activate(-5.0), -5.0);
        assert_eq!(linear.activate(0.0), 0.0);
    }

    #[test]
    fn test_linear_derivative() {
        let linear = ActivationType::Linear;

        // Derivative is always 1
        assert_eq!(linear.derivative(5.0), 1.0);
        assert_eq!(linear.derivative(-5.0), 1.0);
        assert_eq!(linear.derivative(0.0), 1.0);
    }

    #[test]
    fn test_activate_vector() {
        let sigmoid = ActivationType::Sigmoid;
        let input = vec![0.0, 1.0, -1.0];
        let output = sigmoid.activate_vector(&input);

        assert_eq!(output.len(), 3);
        assert_relative_eq!(output[0], 0.5, epsilon = 1e-10);
        assert!(output[1] > 0.7); // sigmoid(1) ≈ 0.731
        assert!(output[2] < 0.3); // sigmoid(-1) ≈ 0.269
    }

    #[test]
    fn test_derivative_vector() {
        let relu = ActivationType::ReLU;
        let input = vec![5.0, -5.0, 0.0];
        let derivs = relu.derivative_vector(&input);

        assert_eq!(derivs.len(), 3);
        assert_eq!(derivs[0], 1.0);
        assert_eq!(derivs[1], 0.0);
        assert_eq!(derivs[2], 0.0);
    }
}
