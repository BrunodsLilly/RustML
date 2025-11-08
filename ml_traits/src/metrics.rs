//! Traits and functions for model evaluation metrics

use crate::Numeric;

/// Trait for regression metrics
pub trait RegressionMetric<T: Numeric> {
    /// Calculate the metric
    ///
    /// # Arguments
    /// * `y_true` - Actual values
    /// * `y_pred` - Predicted values
    ///
    /// # Returns
    /// Metric value
    fn calculate(&self, y_true: &[T], y_pred: &[T]) -> T;

    /// Get the name of the metric
    fn name(&self) -> &str;

    /// Whether higher is better (true) or lower is better (false)
    fn higher_is_better(&self) -> bool;
}

/// Trait for classification metrics
pub trait ClassificationMetric {
    /// Calculate the metric
    ///
    /// # Arguments
    /// * `y_true` - Actual labels
    /// * `y_pred` - Predicted labels
    ///
    /// # Returns
    /// Metric value
    fn calculate(&self, y_true: &[usize], y_pred: &[usize]) -> f64;

    /// Get the name of the metric
    fn name(&self) -> &str;

    /// Whether higher is better
    fn higher_is_better(&self) -> bool {
        true
    }
}

/// Common regression metrics
pub mod regression {
    use super::*;

    /// Mean Squared Error (MSE)
    pub struct MeanSquaredError;

    impl<T: Numeric> RegressionMetric<T> for MeanSquaredError {
        fn calculate(&self, y_true: &[T], y_pred: &[T]) -> T {
            assert_eq!(y_true.len(), y_pred.len());
            let n = T::from_f64(y_true.len() as f64);
            let sum: T = y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(&yt, &yp)| {
                    let diff = yt - yp;
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x);
            sum / n
        }

        fn name(&self) -> &str {
            "Mean Squared Error"
        }

        fn higher_is_better(&self) -> bool {
            false
        }
    }

    /// Root Mean Squared Error (RMSE)
    pub struct RootMeanSquaredError;

    impl<T: Numeric> RegressionMetric<T> for RootMeanSquaredError {
        fn calculate(&self, y_true: &[T], y_pred: &[T]) -> T {
            MeanSquaredError.calculate(y_true, y_pred).sqrt()
        }

        fn name(&self) -> &str {
            "Root Mean Squared Error"
        }

        fn higher_is_better(&self) -> bool {
            false
        }
    }

    /// Mean Absolute Error (MAE)
    pub struct MeanAbsoluteError;

    impl<T: Numeric> RegressionMetric<T> for MeanAbsoluteError {
        fn calculate(&self, y_true: &[T], y_pred: &[T]) -> T {
            assert_eq!(y_true.len(), y_pred.len());
            let n = T::from_f64(y_true.len() as f64);
            let sum: T = y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(&yt, &yp)| (yt - yp).abs())
                .fold(T::zero(), |acc, x| acc + x);
            sum / n
        }

        fn name(&self) -> &str {
            "Mean Absolute Error"
        }

        fn higher_is_better(&self) -> bool {
            false
        }
    }

    /// R² Score (Coefficient of Determination)
    pub struct R2Score;

    impl<T: Numeric> RegressionMetric<T> for R2Score {
        fn calculate(&self, y_true: &[T], y_pred: &[T]) -> T {
            assert_eq!(y_true.len(), y_pred.len());
            let n = T::from_f64(y_true.len() as f64);

            // Calculate mean of y_true
            let y_mean = y_true.iter().fold(T::zero(), |acc, &x| acc + x) / n;

            // Total sum of squares
            let ss_tot: T = y_true
                .iter()
                .map(|&y| {
                    let diff = y - y_mean;
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x);

            // Residual sum of squares
            let ss_res: T = y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(&yt, &yp)| {
                    let diff = yt - yp;
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x);

            T::one() - (ss_res / ss_tot)
        }

        fn name(&self) -> &str {
            "R² Score"
        }

        fn higher_is_better(&self) -> bool {
            true
        }
    }
}

/// Common classification metrics
pub mod classification {
    use super::*;

    /// Accuracy score
    pub struct Accuracy;

    impl ClassificationMetric for Accuracy {
        fn calculate(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
            assert_eq!(y_true.len(), y_pred.len());
            let correct = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(yt, yp)| yt == yp)
                .count();
            correct as f64 / y_true.len() as f64
        }

        fn name(&self) -> &str {
            "Accuracy"
        }
    }

    /// Precision score
    pub struct Precision {
        /// Class to calculate precision for (for binary: 1)
        pub positive_class: usize,
    }

    impl ClassificationMetric for Precision {
        fn calculate(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
            assert_eq!(y_true.len(), y_pred.len());

            let true_positives = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(&yt, &yp)| yt == self.positive_class && yp == self.positive_class)
                .count();

            let predicted_positives = y_pred
                .iter()
                .filter(|&&yp| yp == self.positive_class)
                .count();

            if predicted_positives == 0 {
                0.0
            } else {
                true_positives as f64 / predicted_positives as f64
            }
        }

        fn name(&self) -> &str {
            "Precision"
        }
    }

    /// Recall score
    pub struct Recall {
        /// Class to calculate recall for (for binary: 1)
        pub positive_class: usize,
    }

    impl ClassificationMetric for Recall {
        fn calculate(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
            assert_eq!(y_true.len(), y_pred.len());

            let true_positives = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(&yt, &yp)| yt == self.positive_class && yp == self.positive_class)
                .count();

            let actual_positives = y_true
                .iter()
                .filter(|&&yt| yt == self.positive_class)
                .count();

            if actual_positives == 0 {
                0.0
            } else {
                true_positives as f64 / actual_positives as f64
            }
        }

        fn name(&self) -> &str {
            "Recall"
        }
    }

    /// F1 Score (harmonic mean of precision and recall)
    pub struct F1Score {
        /// Class to calculate F1 for
        pub positive_class: usize,
    }

    impl ClassificationMetric for F1Score {
        fn calculate(&self, y_true: &[usize], y_pred: &[usize]) -> f64 {
            let precision = Precision {
                positive_class: self.positive_class,
            }
            .calculate(y_true, y_pred);

            let recall = Recall {
                positive_class: self.positive_class,
            }
            .calculate(y_true, y_pred);

            if precision + recall == 0.0 {
                0.0
            } else {
                2.0 * (precision * recall) / (precision + recall)
            }
        }

        fn name(&self) -> &str {
            "F1 Score"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use regression::*;
    use classification::*;

    #[test]
    fn test_mse() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.1, 2.1, 2.9, 4.2];
        let mse = MeanSquaredError.calculate(&y_true, &y_pred);
        assert!(mse < 0.1);
    }

    #[test]
    fn test_r2_perfect() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];
        let r2 = R2Score.calculate(&y_true, &y_pred);
        assert!((r2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy() {
        let y_true = vec![0, 1, 2, 2, 1];
        let y_pred = vec![0, 2, 2, 2, 1];
        let acc = Accuracy.calculate(&y_true, &y_pred);
        assert_eq!(acc, 0.8); // 4/5 correct
    }

    #[test]
    fn test_precision_recall() {
        let y_true = vec![1, 1, 0, 1, 0];
        let y_pred = vec![1, 0, 0, 1, 0];

        let precision = Precision { positive_class: 1 }.calculate(&y_true, &y_pred);
        let recall = Recall { positive_class: 1 }.calculate(&y_true, &y_pred);

        assert_eq!(precision, 1.0); // 2/2 predicted positives are correct
        assert_eq!(recall, 2.0 / 3.0); // 2/3 actual positives were found
    }
}
