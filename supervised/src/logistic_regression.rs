//! Logistic Regression classifier
//!
//! Binary and multi-class classification using logistic regression with gradient descent.

use linear_algebra::matrix::Matrix;
use ml_traits::supervised::{Classifier, SupervisedModel};
use ml_traits::Data as DataTrait;

/// Logistic Regression classifier
///
/// Uses gradient descent to learn weights for binary or multi-class classification.
/// For multi-class, uses one-vs-rest (OVR) strategy.
///
/// # Algorithm
/// 1. Initialize weights to zeros
/// 2. For each iteration:
///    - Compute predictions: σ(Xw) where σ is sigmoid
///    - Compute gradient: X^T(y_pred - y_true) / n
///    - Update weights: w -= learning_rate * gradient
/// 3. Repeat until convergence or max iterations
///
/// # Example
/// ```
/// use supervised::LogisticRegression;
/// use linear_algebra::matrix::Matrix;
/// use ml_traits::supervised::SupervisedModel;
///
/// // Binary classification data
/// let X = Matrix::from_vec(
///     vec![
///         1.0, 2.0,
///         2.0, 3.0,
///         3.0, 4.0,
///         6.0, 7.0,
///         7.0, 8.0,
///         8.0, 9.0,
///     ],
///     6,
///     2,
/// ).unwrap();
///
/// let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
///
/// let mut model = LogisticRegression::new(0.1, 100, 1e-4);
/// model.fit(&X, &y).unwrap();
///
/// let predictions = model.predict(&X).unwrap();
/// assert_eq!(predictions.len(), 6);
/// ```
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    /// Learning rate for gradient descent
    learning_rate: f64,
    /// Maximum number of iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Weights for each class (one-vs-rest for multi-class)
    weights: Option<Vec<Vec<f64>>>,
    /// Bias terms for each class
    biases: Option<Vec<f64>>,
    /// Number of classes
    n_classes: usize,
    /// Number of iterations performed
    n_iterations: usize,
    /// Whether the model has been fitted
    fitted: bool,
}

impl LogisticRegression {
    /// Create a new LogisticRegression classifier
    ///
    /// # Arguments
    /// * `learning_rate` - Step size for gradient descent
    /// * `max_iterations` - Maximum training iterations
    /// * `tolerance` - Convergence threshold for weight changes
    pub fn new(learning_rate: f64, max_iterations: usize, tolerance: f64) -> Self {
        assert!(learning_rate > 0.0 && learning_rate.is_finite());
        assert!(max_iterations > 0);
        assert!(tolerance > 0.0 && tolerance.is_finite());

        Self {
            learning_rate,
            max_iterations,
            tolerance,
            weights: None,
            biases: None,
            n_classes: 0,
            n_iterations: 0,
            fitted: false,
        }
    }

    /// Sigmoid activation function
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    /// Train binary classifier
    fn fit_binary(&mut self, X: &Matrix<f64>, y: &[f64]) -> Result<(), String> {
        let n_samples = X.n_samples();
        let n_features = X.n_features();

        // Initialize weights and bias
        let mut weights = vec![0.0; n_features];
        let mut bias = 0.0;

        // Gradient descent
        for iteration in 0..self.max_iterations {
            // Forward pass: compute predictions
            let mut predictions = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let mut z = bias;
                for j in 0..n_features {
                    z += weights[j] * DataTrait::get(X, i, j).unwrap();
                }
                predictions.push(Self::sigmoid(z));
            }

            // Compute gradients
            let mut weight_gradients = vec![0.0; n_features];
            let mut bias_gradient = 0.0;

            for i in 0..n_samples {
                let error = predictions[i] - y[i];
                bias_gradient += error;
                for j in 0..n_features {
                    weight_gradients[j] += error * DataTrait::get(X, i, j).unwrap();
                }
            }

            // Average gradients
            let n = n_samples as f64;
            for grad in &mut weight_gradients {
                *grad /= n;
            }
            bias_gradient /= n;

            // Update parameters
            let mut max_change: f64 = 0.0;
            for j in 0..n_features {
                let change = self.learning_rate * weight_gradients[j];
                weights[j] -= change;
                max_change = max_change.max(change.abs());
            }
            let bias_change = self.learning_rate * bias_gradient;
            bias -= bias_change;
            max_change = max_change.max(bias_change.abs());

            self.n_iterations = iteration + 1;

            // Check convergence
            if max_change < self.tolerance {
                break;
            }
        }

        self.weights = Some(vec![weights]);
        self.biases = Some(vec![bias]);
        self.n_classes = 2;

        Ok(())
    }

    /// Train multi-class classifier using one-vs-rest
    fn fit_multiclass(&mut self, X: &Matrix<f64>, y: &[f64]) -> Result<(), String> {
        let n_samples = X.n_samples();
        let n_features = X.n_features();

        // Find unique classes
        let mut classes: Vec<usize> = y.iter().map(|&v| v as usize).collect();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();

        let mut all_weights = Vec::with_capacity(n_classes);
        let mut all_biases = Vec::with_capacity(n_classes);

        // Train one classifier per class (one-vs-rest)
        for &class_label in &classes {
            // Create binary labels (1 for current class, 0 for others)
            let binary_y: Vec<f64> = y.iter().map(|&label| {
                if (label as usize) == class_label { 1.0 } else { 0.0 }
            }).collect();

            // Train binary classifier for this class
            let mut temp_model = LogisticRegression::new(
                self.learning_rate,
                self.max_iterations,
                self.tolerance,
            );
            temp_model.fit_binary(X, &binary_y)?;

            all_weights.push(temp_model.weights.unwrap()[0].clone());
            all_biases.push(temp_model.biases.unwrap()[0]);
        }

        self.weights = Some(all_weights);
        self.biases = Some(all_biases);
        self.n_classes = n_classes;

        Ok(())
    }

    /// Predict probabilities for each class
    pub fn predict_proba(&self, X: &Matrix<f64>) -> Result<Vec<Vec<f64>>, String> {
        if !self.is_fitted() {
            return Err("Model not fitted. Call fit() first.".to_string());
        }

        let n_samples = X.n_samples();
        let n_features = X.n_features();
        let weights = self.weights.as_ref().unwrap();
        let biases = self.biases.as_ref().unwrap();

        if n_features != weights[0].len() {
            return Err(format!(
                "X has {} features, but model was trained with {} features",
                n_features,
                weights[0].len()
            ));
        }

        let mut probabilities = vec![vec![0.0; self.n_classes]; n_samples];

        if self.n_classes == 2 {
            // Binary classification
            for i in 0..n_samples {
                let mut z = biases[0];
                for j in 0..n_features {
                    z += weights[0][j] * DataTrait::get(X, i, j).unwrap();
                }
                let prob = Self::sigmoid(z);
                probabilities[i][0] = 1.0 - prob;
                probabilities[i][1] = prob;
            }
        } else {
            // Multi-class: one-vs-rest
            for i in 0..n_samples {
                let mut class_scores = vec![0.0; self.n_classes];

                for (class_idx, (w, &b)) in weights.iter().zip(biases.iter()).enumerate() {
                    let mut z = b;
                    for j in 0..n_features {
                        z += w[j] * DataTrait::get(X, i, j).unwrap();
                    }
                    class_scores[class_idx] = Self::sigmoid(z);
                }

                // Normalize to sum to 1
                let sum: f64 = class_scores.iter().sum();
                if sum > 0.0 {
                    for score in &mut class_scores {
                        *score /= sum;
                    }
                }

                probabilities[i] = class_scores;
            }
        }

        Ok(probabilities)
    }

    /// Get model weights
    pub fn weights(&self) -> Option<&Vec<Vec<f64>>> {
        self.weights.as_ref()
    }

    /// Get model biases
    pub fn biases(&self) -> Option<&Vec<f64>> {
        self.biases.as_ref()
    }
}

impl SupervisedModel<f64, Matrix<f64>> for LogisticRegression {
    type Prediction = Vec<usize>;

    fn fit(&mut self, X: &Matrix<f64>, y: &[f64]) -> Result<(), String> {
        if X.n_samples() != y.len() {
            return Err(format!(
                "X has {} samples but y has {} labels",
                X.n_samples(),
                y.len()
            ));
        }

        if X.n_samples() == 0 {
            return Err("Cannot fit on empty data".to_string());
        }

        // Determine number of classes
        let mut unique_labels: Vec<usize> = y.iter().map(|&v| v as usize).collect();
        unique_labels.sort_unstable();
        unique_labels.dedup();
        let n_classes = unique_labels.len();

        if n_classes < 2 {
            return Err("Need at least 2 classes for classification".to_string());
        }

        if n_classes == 2 {
            self.fit_binary(X, y)
        } else {
            self.fit_multiclass(X, y)
        }?;

        self.fitted = true;
        Ok(())
    }

    fn predict(&self, X: &Matrix<f64>) -> Result<Vec<usize>, String> {
        let probas = self.predict_proba(X)?;

        // Return class with highest probability
        Ok(probas
            .iter()
            .map(|probs| {
                probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            })
            .collect())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl Classifier<f64, Matrix<f64>> for LogisticRegression {
    fn n_classes(&self) -> usize {
        self.n_classes
    }

    fn predict_proba(&self, X: &Matrix<f64>) -> Result<Vec<Vec<f64>>, String> {
        self.predict_proba(X)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_classification() {
        // Linearly separable binary classification
        let X = Matrix::from_vec(
            vec![
                1.0, 2.0,
                2.0, 3.0,
                3.0, 4.0,
                6.0, 7.0,
                7.0, 8.0,
                8.0, 9.0,
            ],
            6,
            2,
        )
        .unwrap();

        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut model = LogisticRegression::new(0.1, 1000, 1e-4);
        model.fit(&X, &y).unwrap();

        let predictions = model.predict(&X).unwrap();
        assert_eq!(predictions.len(), 6);

        // Should correctly classify all points
        assert_eq!(predictions[0], 0);
        assert_eq!(predictions[1], 0);
        assert_eq!(predictions[2], 0);
        assert_eq!(predictions[3], 1);
        assert_eq!(predictions[4], 1);
        assert_eq!(predictions[5], 1);
    }

    #[test]
    fn test_predict_proba() {
        let X = Matrix::from_vec(vec![1.0, 2.0, 7.0, 8.0], 2, 2).unwrap();
        let y = vec![0.0, 1.0];

        let mut model = LogisticRegression::new(0.1, 1000, 1e-4);
        model.fit(&X, &y).unwrap();

        let probas = model.predict_proba(&X).unwrap();
        assert_eq!(probas.len(), 2);
        assert_eq!(probas[0].len(), 2);

        // Probabilities should sum to ~1
        for probs in probas {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_multiclass_classification() {
        // Three-class problem
        let X = Matrix::from_vec(
            vec![
                1.0, 1.0,
                2.0, 1.0,
                5.0, 5.0,
                6.0, 5.0,
                9.0, 9.0,
                10.0, 9.0,
            ],
            6,
            2,
        )
        .unwrap();

        let y = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let mut model = LogisticRegression::new(0.1, 1000, 1e-4);
        model.fit(&X, &y).unwrap();

        assert_eq!(model.n_classes(), 3);

        let predictions = model.predict(&X).unwrap();
        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_not_fitted() {
        let model = LogisticRegression::new(0.1, 100, 1e-4);
        let X = Matrix::from_vec(vec![1.0, 2.0], 1, 2).unwrap();

        let result = model.predict(&X);
        assert!(result.is_err());
    }

    #[test]
    fn test_sigmoid() {
        assert!((LogisticRegression::sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(LogisticRegression::sigmoid(10.0) > 0.99);
        assert!(LogisticRegression::sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_convergence() {
        let X = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let y = vec![0.0, 1.0];

        let mut model = LogisticRegression::new(0.5, 1000, 1e-6);
        model.fit(&X, &y).unwrap();

        // Model should complete training
        assert!(model.n_iterations > 0);
        assert!(model.is_fitted());
    }

    #[test]
    fn test_mismatched_dimensions() {
        let X = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let y = vec![0.0]; // Wrong size

        let mut model = LogisticRegression::new(0.1, 100, 1e-4);
        let result = model.fit(&X, &y);
        assert!(result.is_err());
    }
}
