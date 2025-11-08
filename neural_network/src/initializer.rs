use linear_algebra::{matrix::Matrix, vectors::Vector};
use rand::distributions::{Distribution, Uniform};

/// Weight initialization strategies for neural networks
#[derive(Debug, Clone, Copy)]
pub enum Initializer {
    /// Xavier/Glorot initialization
    ///
    /// Weights drawn from Uniform(-limit, limit) where:
    /// limit = sqrt(6 / (fan_in + fan_out))
    ///
    /// Good for: Sigmoid, Tanh activations
    Xavier,

    /// He initialization
    ///
    /// Weights drawn from Normal(0, sqrt(2 / fan_in))
    ///
    /// Good for: ReLU activation
    He,

    /// Zero initialization (for biases)
    Zeros,

    /// Small random initialization
    ///
    /// Weights drawn from Uniform(-0.01, 0.01)
    SmallRandom,
}

impl Initializer {
    /// Initialize a weight matrix
    ///
    /// # Arguments
    /// * `rows` - Number of rows (output neurons)
    /// * `cols` - Number of columns (input neurons)
    ///
    /// # Returns
    /// Matrix with initialized weights
    pub fn initialize_matrix(&self, rows: usize, cols: usize) -> Matrix<f64> {
        let mut rng = rand::thread_rng();

        match self {
            Initializer::Xavier => {
                let fan_in = cols as f64;
                let fan_out = rows as f64;
                let limit = (6.0 / (fan_in + fan_out)).sqrt();
                let dist = Uniform::new(-limit, limit);

                let data: Vec<f64> = (0..rows * cols).map(|_| dist.sample(&mut rng)).collect();

                Matrix::from_vec(data, rows, cols).unwrap()
            }
            Initializer::He => {
                let fan_in = cols as f64;
                let std_dev = (2.0 / fan_in).sqrt();

                // Approximate normal distribution with uniform
                // Using uniform [-sqrt(3) * std_dev, sqrt(3) * std_dev]
                // This gives same variance as Normal(0, std_dev)
                let limit = 3.0_f64.sqrt() * std_dev;
                let dist = Uniform::new(-limit, limit);

                let data: Vec<f64> = (0..rows * cols).map(|_| dist.sample(&mut rng)).collect();

                Matrix::from_vec(data, rows, cols).unwrap()
            }
            Initializer::Zeros => Matrix::zeros(rows, cols),
            Initializer::SmallRandom => {
                let dist = Uniform::new(-0.01, 0.01);
                let data: Vec<f64> = (0..rows * cols).map(|_| dist.sample(&mut rng)).collect();

                Matrix::from_vec(data, rows, cols).unwrap()
            }
        }
    }

    /// Initialize a bias vector
    ///
    /// # Arguments
    /// * `size` - Number of neurons (bias values)
    ///
    /// # Returns
    /// Vector with initialized biases (typically zeros)
    pub fn initialize_vector(&self, size: usize) -> Vector<f64> {
        match self {
            Initializer::Zeros => Vector {
                data: vec![0.0; size],
            },
            _ => {
                // Biases typically initialized to zero regardless of strategy
                Vector {
                    data: vec![0.0; size],
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xavier_initialization() {
        let init = Initializer::Xavier;
        let weights = init.initialize_matrix(10, 20);

        assert_eq!(weights.rows, 10);
        assert_eq!(weights.cols, 20);

        // Check that weights are within expected range
        let fan_in = 20.0_f64;
        let fan_out = 10.0_f64;
        let limit = (6.0_f64 / (fan_in + fan_out)).sqrt();

        for &w in &weights.data {
            assert!(
                w >= -limit && w <= limit,
                "Weight {} outside range [-{}, {}]",
                w,
                limit,
                limit
            );
        }

        // Check variance is roughly correct
        let mean: f64 = weights.data.iter().sum::<f64>() / weights.data.len() as f64;
        let variance: f64 = weights
            .data
            .iter()
            .map(|&w| (w - mean).powi(2))
            .sum::<f64>()
            / weights.data.len() as f64;

        let expected_variance = 2.0 / (fan_in + fan_out);

        // Variance should be within 50% of expected (statistical test)
        assert!(
            variance > expected_variance * 0.5 && variance < expected_variance * 1.5,
            "Variance {} not close to expected {}",
            variance,
            expected_variance
        );
    }

    #[test]
    fn test_he_initialization() {
        let init = Initializer::He;
        let weights = init.initialize_matrix(10, 20);

        assert_eq!(weights.rows, 10);
        assert_eq!(weights.cols, 20);

        // Check variance is roughly correct
        let mean: f64 = weights.data.iter().sum::<f64>() / weights.data.len() as f64;
        let variance: f64 = weights
            .data
            .iter()
            .map(|&w| (w - mean).powi(2))
            .sum::<f64>()
            / weights.data.len() as f64;

        let fan_in = 20.0;
        let expected_variance = 2.0 / fan_in;

        // Variance should be within 50% of expected
        assert!(
            variance > expected_variance * 0.5 && variance < expected_variance * 1.5,
            "Variance {} not close to expected {}",
            variance,
            expected_variance
        );
    }

    #[test]
    fn test_zeros_initialization() {
        let init = Initializer::Zeros;
        let weights = init.initialize_matrix(5, 10);

        assert_eq!(weights.rows, 5);
        assert_eq!(weights.cols, 10);

        // All weights should be zero
        for &w in &weights.data {
            assert_eq!(w, 0.0);
        }
    }

    #[test]
    fn test_small_random_initialization() {
        let init = Initializer::SmallRandom;
        let weights = init.initialize_matrix(5, 10);

        assert_eq!(weights.rows, 5);
        assert_eq!(weights.cols, 10);

        // All weights should be in range [-0.01, 0.01]
        for &w in &weights.data {
            assert!(w >= -0.01 && w <= 0.01);
        }
    }

    #[test]
    fn test_bias_initialization() {
        let init = Initializer::Xavier;
        let biases = init.initialize_vector(10);

        assert_eq!(biases.data.len(), 10);

        // Biases should be zero
        for &b in &biases.data {
            assert_eq!(b, 0.0);
        }
    }

    #[test]
    fn test_different_runs_produce_different_weights() {
        let init = Initializer::Xavier;
        let weights1 = init.initialize_matrix(5, 5);
        let weights2 = init.initialize_matrix(5, 5);

        // Extremely unlikely to get identical random matrices
        assert_ne!(weights1.data, weights2.data);
    }
}
