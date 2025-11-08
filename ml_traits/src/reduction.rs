//! Traits specific to dimensionality reduction algorithms

use crate::{Data, Numeric};
use crate::unsupervised::DimensionalityReduction;

/// Trait for linear dimensionality reduction methods (PCA, etc.)
pub trait LinearReduction<T: Numeric, D: Data<T>>: DimensionalityReduction<T, D> {
    /// Get the singular values
    fn singular_values(&self) -> Option<Vec<T>> {
        None
    }

    /// Get the noise variance (if estimated)
    fn noise_variance(&self) -> Option<T> {
        None
    }
}

/// Trait for non-linear dimensionality reduction (t-SNE, UMAP, etc.)
pub trait NonLinearReduction<T: Numeric, D: Data<T>>: DimensionalityReduction<T, D> {
    /// Get the KL divergence (for t-SNE)
    fn kl_divergence(&self) -> Option<T> {
        None
    }

    /// Get the perplexity parameter (for t-SNE)
    fn perplexity(&self) -> Option<T> {
        None
    }

    /// Get number of iterations run
    fn n_iter(&self) -> usize {
        0
    }
}
