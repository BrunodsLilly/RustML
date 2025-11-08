//! Traits for unsupervised learning algorithms

use crate::{Data, Numeric};

/// Core trait for unsupervised learning models
///
/// Unsupervised models find patterns in data without labels.
pub trait UnsupervisedModel<T: Numeric, D: Data<T>> {
    /// Train the model on unlabeled data
    ///
    /// # Arguments
    /// * `X` - Training features (no labels)
    ///
    /// # Returns
    /// Result indicating success or error
    fn fit(&mut self, X: &D) -> Result<(), String>;

    /// Transform data using the fitted model
    ///
    /// # Arguments
    /// * `X` - Data to transform
    ///
    /// # Returns
    /// Transformed data
    fn transform(&self, X: &D) -> Result<Vec<Vec<T>>, String>;

    /// Fit and transform in one step
    ///
    /// # Arguments
    /// * `X` - Data to fit and transform
    ///
    /// # Returns
    /// Transformed data
    fn fit_transform(&mut self, X: &D) -> Result<Vec<Vec<T>>, String> {
        self.fit(X)?;
        self.transform(X)
    }

    /// Check if the model has been fitted
    fn is_fitted(&self) -> bool;
}

/// Trait for dimensionality reduction algorithms
///
/// These models reduce the number of features while preserving important information.
pub trait DimensionalityReduction<T: Numeric, D: Data<T>>: UnsupervisedModel<T, D> {
    /// Get the number of output dimensions
    fn n_components(&self) -> usize;

    /// Get explained variance ratio (if applicable)
    fn explained_variance_ratio(&self) -> Option<Vec<T>> {
        None
    }

    /// Get the components/loadings (if applicable)
    fn components(&self) -> Option<Vec<Vec<T>>> {
        None
    }
}

/// Trait for manifold learning algorithms
///
/// These algorithms learn low-dimensional representations of high-dimensional data.
pub trait ManifoldLearner<T: Numeric, D: Data<T>>: UnsupervisedModel<T, D> {
    /// Get the embedding dimension
    fn embedding_dim(&self) -> usize;

    /// Check if the algorithm is iterative
    fn is_iterative(&self) -> bool {
        false
    }

    /// Get current iteration (for iterative algorithms)
    fn current_iteration(&self) -> Option<usize> {
        None
    }
}
