/// Structured error types for ML algorithms
///
/// This module provides comprehensive error handling for machine learning operations.
/// All ML algorithms should use these error types instead of String-based errors.
use std::fmt;

/// Common error types for ML algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum MLError {
    /// Invalid input parameters
    InvalidInput {
        message: String,
        parameter: &'static str,
    },

    /// Model has not been fitted before prediction
    NotFitted { model_type: &'static str },

    /// Dimension mismatch between expected and provided data
    DimensionMismatch {
        expected: (usize, usize),
        got: (usize, usize),
        context: String,
    },

    /// Algorithm failed to converge
    ConvergenceFailure {
        iterations: usize,
        final_cost: f64,
        threshold: f64,
    },

    /// Insufficient data for the operation
    InsufficientData {
        required: usize,
        provided: usize,
        operation: &'static str,
    },

    /// Numerical instability detected
    NumericalInstability { context: String, value: f64 },

    /// Matrix operation error
    MatrixError {
        operation: &'static str,
        reason: String,
    },

    /// CSV/Data loading error
    DataLoadError { source: String, details: String },

    /// Invalid hyperparameter value
    InvalidHyperparameter {
        name: &'static str,
        value: String,
        constraint: String,
    },

    /// Algorithm-specific error
    AlgorithmError {
        algorithm: &'static str,
        message: String,
    },
}

impl fmt::Display for MLError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MLError::InvalidInput { message, parameter } => {
                write!(
                    f,
                    "Invalid input for parameter '{}': {}",
                    parameter, message
                )
            }
            MLError::NotFitted { model_type } => {
                write!(
                    f,
                    "{} model has not been fitted. Call fit() before predict().",
                    model_type
                )
            }
            MLError::DimensionMismatch {
                expected,
                got,
                context,
            } => {
                write!(
                    f,
                    "Dimension mismatch in {}: expected {}x{}, got {}x{}",
                    context, expected.0, expected.1, got.0, got.1
                )
            }
            MLError::ConvergenceFailure {
                iterations,
                final_cost,
                threshold,
            } => {
                write!(
                    f,
                    "Failed to converge after {} iterations (final cost: {:.6}, threshold: {:.6})",
                    iterations, final_cost, threshold
                )
            }
            MLError::InsufficientData {
                required,
                provided,
                operation,
            } => {
                write!(
                    f,
                    "Insufficient data for {}: requires {} samples, got {}",
                    operation, required, provided
                )
            }
            MLError::NumericalInstability { context, value } => {
                write!(
                    f,
                    "Numerical instability detected in {}: value = {}",
                    context, value
                )
            }
            MLError::MatrixError { operation, reason } => {
                write!(f, "Matrix operation '{}' failed: {}", operation, reason)
            }
            MLError::DataLoadError { source, details } => {
                write!(f, "Failed to load data from '{}': {}", source, details)
            }
            MLError::InvalidHyperparameter {
                name,
                value,
                constraint,
            } => {
                write!(
                    f,
                    "Invalid hyperparameter '{}' = '{}': must satisfy {}",
                    name, value, constraint
                )
            }
            MLError::AlgorithmError { algorithm, message } => {
                write!(f, "{} error: {}", algorithm, message)
            }
        }
    }
}

impl std::error::Error for MLError {}

// Conversion from String for backwards compatibility during migration
impl From<String> for MLError {
    fn from(s: String) -> Self {
        MLError::AlgorithmError {
            algorithm: "Unknown",
            message: s,
        }
    }
}

impl From<&str> for MLError {
    fn from(s: &str) -> Self {
        MLError::AlgorithmError {
            algorithm: "Unknown",
            message: s.to_string(),
        }
    }
}

/// Helper functions for creating common errors
impl MLError {
    /// Create an invalid input error
    pub fn invalid_input(parameter: &'static str, message: impl Into<String>) -> Self {
        MLError::InvalidInput {
            message: message.into(),
            parameter,
        }
    }

    /// Create a not fitted error
    pub fn not_fitted(model_type: &'static str) -> Self {
        MLError::NotFitted { model_type }
    }

    /// Create a dimension mismatch error
    pub fn dimension_mismatch(
        expected: (usize, usize),
        got: (usize, usize),
        context: impl Into<String>,
    ) -> Self {
        MLError::DimensionMismatch {
            expected,
            got,
            context: context.into(),
        }
    }

    /// Create a convergence failure error
    pub fn convergence_failure(iterations: usize, final_cost: f64, threshold: f64) -> Self {
        MLError::ConvergenceFailure {
            iterations,
            final_cost,
            threshold,
        }
    }

    /// Create an insufficient data error
    pub fn insufficient_data(required: usize, provided: usize, operation: &'static str) -> Self {
        MLError::InsufficientData {
            required,
            provided,
            operation,
        }
    }

    /// Create a numerical instability error
    pub fn numerical_instability(context: impl Into<String>, value: f64) -> Self {
        MLError::NumericalInstability {
            context: context.into(),
            value,
        }
    }

    /// Create a matrix error
    pub fn matrix_error(operation: &'static str, reason: impl Into<String>) -> Self {
        MLError::MatrixError {
            operation,
            reason: reason.into(),
        }
    }

    /// Create a data load error
    pub fn data_load_error(source: impl Into<String>, details: impl Into<String>) -> Self {
        MLError::DataLoadError {
            source: source.into(),
            details: details.into(),
        }
    }

    /// Create an invalid hyperparameter error
    pub fn invalid_hyperparameter(
        name: &'static str,
        value: impl Into<String>,
        constraint: impl Into<String>,
    ) -> Self {
        MLError::InvalidHyperparameter {
            name,
            value: value.into(),
            constraint: constraint.into(),
        }
    }

    /// Create an algorithm-specific error
    pub fn algorithm_error(algorithm: &'static str, message: impl Into<String>) -> Self {
        MLError::AlgorithmError {
            algorithm,
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_input_error() {
        let err = MLError::invalid_input("learning_rate", "must be positive");
        assert_eq!(
            err.to_string(),
            "Invalid input for parameter 'learning_rate': must be positive"
        );
    }

    #[test]
    fn test_not_fitted_error() {
        let err = MLError::not_fitted("KMeans");
        assert_eq!(
            err.to_string(),
            "KMeans model has not been fitted. Call fit() before predict()."
        );
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let err = MLError::dimension_mismatch((10, 5), (10, 3), "feature matrix");
        assert!(err.to_string().contains("expected 10x5, got 10x3"));
    }

    #[test]
    fn test_convergence_failure_error() {
        let err = MLError::convergence_failure(100, 0.5, 1e-4);
        assert!(err
            .to_string()
            .contains("Failed to converge after 100 iterations"));
    }

    #[test]
    fn test_insufficient_data_error() {
        let err = MLError::insufficient_data(10, 5, "K-Means clustering");
        assert_eq!(
            err.to_string(),
            "Insufficient data for K-Means clustering: requires 10 samples, got 5"
        );
    }

    #[test]
    fn test_numerical_instability_error() {
        let err = MLError::numerical_instability("gradient computation", f64::NAN);
        assert!(err.to_string().contains("Numerical instability"));
    }

    #[test]
    fn test_from_string() {
        let err: MLError = "test error".to_string().into();
        assert!(matches!(err, MLError::AlgorithmError { .. }));
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_helper_methods() {
        let err = MLError::invalid_hyperparameter("k", "0", "k > 0");
        let err_string = err.to_string();
        assert!(
            err_string.contains("'k'"),
            "Expected to find 'k' in: {}",
            err_string
        );
        assert!(
            err_string.contains("'0'"),
            "Expected to find '0' in: {}",
            err_string
        );
        assert!(
            err_string.contains("must satisfy k > 0"),
            "Expected to find 'must satisfy k > 0' in: {}",
            err_string
        );
    }
}
