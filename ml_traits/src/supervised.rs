//! Traits for supervised learning algorithms

use crate::{Data, Numeric};

/// Core trait for supervised learning models
///
/// This trait defines the interface that all supervised learning algorithms must implement.
/// It enables generic code that works with any model type (regression, classification, etc.)
///
/// # Type Parameters
/// * `T` - Numeric type (f32, f64, etc.)
/// * `D` - Data type (Matrix, DataFrame, etc.)
pub trait SupervisedModel<T: Numeric, D: Data<T>> {
    /// The type of predictions this model produces
    type Prediction;

    /// Train the model on labeled data
    ///
    /// # Arguments
    /// * `X` - Training features
    /// * `y` - Training labels/targets
    ///
    /// # Returns
    /// Result indicating success or error message
    fn fit(&mut self, X: &D, y: &[T]) -> Result<(), String>;

    /// Make predictions on new data
    ///
    /// # Arguments
    /// * `X` - Features to predict
    ///
    /// # Returns
    /// Predictions for each sample
    fn predict(&self, X: &D) -> Result<Self::Prediction, String>;

    /// Check if the model has been trained
    fn is_fitted(&self) -> bool;
}

/// Trait for regression models
///
/// Regression models predict continuous values.
pub trait Regressor<T: Numeric, D: Data<T>>: SupervisedModel<T, D, Prediction = Vec<T>> {
    /// Get the model's learned parameters (if applicable)
    fn parameters(&self) -> Option<Vec<T>> {
        None
    }

    /// Predict a single value
    fn predict_single(&self, features: &[T]) -> Result<T, String>;
}

/// Trait for classification models
///
/// Classification models predict discrete classes.
pub trait Classifier<T: Numeric, D: Data<T>>: SupervisedModel<T, D, Prediction = Vec<usize>> {
    /// Get the number of classes
    fn n_classes(&self) -> usize;

    /// Predict class probabilities (if supported)
    fn predict_proba(&self, X: &D) -> Result<Vec<Vec<T>>, String> {
        Err("This model does not support probability predictions".to_string())
    }

    /// Get class labels
    fn classes(&self) -> Vec<usize> {
        (0..self.n_classes()).collect()
    }
}

/// Trait for models that support incremental learning
///
/// These models can be trained on batches of data without retraining from scratch.
pub trait IncrementalLearner<T: Numeric, D: Data<T>>: SupervisedModel<T, D> {
    /// Partially fit the model on a batch of data
    ///
    /// # Arguments
    /// * `X` - Batch features
    /// * `y` - Batch labels
    ///
    /// # Returns
    /// Result indicating success or error
    fn partial_fit(&mut self, X: &D, y: &[T]) -> Result<(), String>;
}

// Tests will be added when concrete implementations are created
