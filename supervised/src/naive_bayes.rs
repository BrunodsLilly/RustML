//! Naive Bayes Classifier
//!
//! Implements Gaussian Naive Bayes for classification based on Bayes' theorem
//! with the "naive" assumption of feature independence.

use linear_algebra::matrix::Matrix;
use ml_traits::supervised::SupervisedModel;
use ml_traits::Data;
use std::collections::HashMap;

/// Gaussian Naive Bayes Classifier
///
/// Assumes features follow a Gaussian (normal) distribution within each class.
/// Computes mean and variance for each feature per class during training.
///
/// # Algorithm
/// 1. Calculate prior probabilities P(class) from training data
/// 2. For each class and feature, calculate mean μ and variance σ²
/// 3. For prediction, use Bayes' theorem with Gaussian likelihood
///
/// # Example
/// ```
/// use supervised::GaussianNB;
/// use linear_algebra::matrix::Matrix;
/// use ml_traits::supervised::SupervisedModel;
///
/// let X = Matrix::from_vec(
///     vec![
///         1.0, 2.0,  // Class 0
///         2.0, 3.0,  // Class 0
///         8.0, 9.0,  // Class 1
///         9.0, 10.0, // Class 1
///     ],
///     4, 2
/// ).unwrap();
/// let y = vec![0.0, 0.0, 1.0, 1.0];
///
/// let mut nb = GaussianNB::new();
/// nb.fit(&X, &y).unwrap();
/// let predictions = nb.predict(&X).unwrap();
/// assert_eq!(predictions, vec![0, 0, 1, 1]);
/// ```
pub struct GaussianNB {
    /// Class priors P(class)
    class_priors: HashMap<usize, f64>,
    /// Mean μ for each class and feature
    class_means: HashMap<usize, Vec<f64>>,
    /// Variance σ² for each class and feature
    class_variances: HashMap<usize, Vec<f64>>,
    /// Number of classes
    n_classes: usize,
    /// Number of features
    n_features: usize,
    /// Small constant for numerical stability
    epsilon: f64,
    /// Whether the model has been fitted
    fitted: bool,
}

impl GaussianNB {
    /// Create a new Gaussian Naive Bayes classifier
    pub fn new() -> Self {
        Self {
            class_priors: HashMap::new(),
            class_means: HashMap::new(),
            class_variances: HashMap::new(),
            n_classes: 0,
            n_features: 0,
            epsilon: 1e-9,
            fitted: false,
        }
    }

    /// Calculate Gaussian probability density
    fn gaussian_pdf(x: f64, mean: f64, variance: f64) -> f64 {
        let std = variance.sqrt();
        let coefficient = 1.0 / (std * (2.0 * std::f64::consts::PI).sqrt());
        let exponent = -((x - mean).powi(2)) / (2.0 * variance);
        coefficient * exponent.exp()
    }

    /// Calculate log posterior probability for a class
    fn log_posterior(&self, sample: &[f64], class: usize) -> f64 {
        let prior = self.class_priors.get(&class).unwrap();
        let means = self.class_means.get(&class).unwrap();
        let variances = self.class_variances.get(&class).unwrap();

        // Start with log prior
        let mut log_prob = prior.ln();

        // Add log likelihood for each feature (sum in log space = product in prob space)
        for (i, &x) in sample.iter().enumerate() {
            let pdf = Self::gaussian_pdf(x, means[i], variances[i]);
            log_prob += pdf.max(self.epsilon).ln();
        }

        log_prob
    }
}

impl Default for GaussianNB {
    fn default() -> Self {
        Self::new()
    }
}

impl SupervisedModel<f64, Matrix<f64>> for GaussianNB {
    type Prediction = Vec<usize>;

    fn fit(&mut self, X: &Matrix<f64>, y: &[f64]) -> Result<(), String> {
        if X.n_samples() != y.len() {
            return Err(format!(
                "X has {} samples but y has {} labels",
                X.n_samples(),
                y.len()
            ));
        }

        self.n_features = X.n_features();

        // Group samples by class
        let mut class_samples: HashMap<usize, Vec<Vec<f64>>> = HashMap::new();
        for i in 0..X.n_samples() {
            let class = y[i] as usize;
            let sample: Vec<f64> = (0..X.n_features())
                .map(|j| *X.get(i, j).expect("Valid index"))
                .collect();
            class_samples
                .entry(class)
                .or_insert_with(Vec::new)
                .push(sample);
        }

        self.n_classes = class_samples.len();
        let total_samples = X.n_samples() as f64;

        // Calculate priors, means, and variances for each class
        for (class, samples) in class_samples.iter() {
            let n_class_samples = samples.len() as f64;

            // Prior probability
            self.class_priors
                .insert(*class, n_class_samples / total_samples);

            // Calculate mean for each feature
            let mut means = vec![0.0; self.n_features];
            for sample in samples {
                for (j, &value) in sample.iter().enumerate() {
                    means[j] += value;
                }
            }
            for mean in &mut means {
                *mean /= n_class_samples;
            }

            // Calculate variance for each feature
            let mut variances = vec![0.0; self.n_features];
            for sample in samples {
                for (j, &value) in sample.iter().enumerate() {
                    let diff = value - means[j];
                    variances[j] += diff * diff;
                }
            }
            for variance in &mut variances {
                *variance /= n_class_samples;
                *variance += self.epsilon; // Add small constant for numerical stability
            }

            self.class_means.insert(*class, means);
            self.class_variances.insert(*class, variances);
        }

        self.fitted = true;
        Ok(())
    }

    fn predict(&self, X: &Matrix<f64>) -> Result<Self::Prediction, String> {
        if !self.fitted {
            return Err("Model not fitted. Call fit() first.".to_string());
        }

        let mut predictions = Vec::with_capacity(X.n_samples());

        for i in 0..X.n_samples() {
            let sample: Vec<f64> = (0..X.n_features())
                .map(|j| *X.get(i, j).expect("Valid index"))
                .collect();

            // Find class with maximum log posterior
            let mut best_class = 0;
            let mut best_log_prob = f64::NEG_INFINITY;

            for class in 0..self.n_classes {
                let log_prob = self.log_posterior(&sample, class);
                if log_prob > best_log_prob {
                    best_log_prob = log_prob;
                    best_class = class;
                }
            }

            predictions.push(best_class);
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_nb_basic() {
        // Simple separable data
        let X = Matrix::from_vec(
            vec![
                1.0, 2.0, // Class 0
                2.0, 3.0, // Class 0
                8.0, 9.0, // Class 1
                9.0, 10.0, // Class 1
            ],
            4,
            2,
        )
        .unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];

        let mut nb = GaussianNB::new();
        nb.fit(&X, &y).unwrap();

        let predictions = nb.predict(&X).unwrap();
        assert_eq!(predictions, vec![0, 0, 1, 1]);
    }

    #[test]
    fn test_gaussian_nb_not_fitted() {
        let nb = GaussianNB::new();
        let X = Matrix::from_vec(vec![1.0, 2.0], 1, 2).unwrap();

        let result = nb.predict(&X);
        assert!(result.is_err());
    }

    #[test]
    fn test_gaussian_nb_multiclass() {
        // Three classes
        let X = Matrix::from_vec(
            vec![
                1.0, 1.0, // Class 0
                2.0, 2.0, // Class 0
                5.0, 5.0, // Class 1
                6.0, 6.0, // Class 1
                9.0, 9.0, // Class 2
                10.0, 10.0, // Class 2
            ],
            6,
            2,
        )
        .unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let mut nb = GaussianNB::new();
        nb.fit(&X, &y).unwrap();

        let predictions = nb.predict(&X).unwrap();
        assert_eq!(predictions, vec![0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn test_gaussian_pdf() {
        // Test Gaussian probability density function
        let pdf = GaussianNB::gaussian_pdf(0.0, 0.0, 1.0);
        // For standard normal, pdf(0) = 1/sqrt(2π) ≈ 0.3989
        assert!((pdf - 0.3989).abs() < 0.001);
    }
}
