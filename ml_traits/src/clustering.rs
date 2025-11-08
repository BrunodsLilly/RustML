//! Traits for clustering algorithms

use crate::{Data, Numeric};

/// Core trait for clustering algorithms
///
/// Clustering algorithms group similar data points together.
pub trait Clusterer<T: Numeric, D: Data<T>> {
    /// Fit the clustering model
    ///
    /// # Arguments
    /// * `X` - Data to cluster
    ///
    /// # Returns
    /// Result indicating success or error
    fn fit(&mut self, X: &D) -> Result<(), String>;

    /// Predict cluster labels for data
    ///
    /// # Arguments
    /// * `X` - Data to assign to clusters
    ///
    /// # Returns
    /// Cluster label for each sample
    fn predict(&self, X: &D) -> Result<Vec<usize>, String>;

    /// Fit and predict in one step
    ///
    /// # Arguments
    /// * `X` - Data to cluster
    ///
    /// # Returns
    /// Cluster labels
    fn fit_predict(&mut self, X: &D) -> Result<Vec<usize>, String> {
        self.fit(X)?;
        self.predict(X)
    }

    /// Get the number of clusters
    fn n_clusters(&self) -> usize;

    /// Check if the model has been fitted
    fn is_fitted(&self) -> bool;
}

/// Trait for centroid-based clustering algorithms (K-means, etc.)
pub trait CentroidClusterer<T: Numeric, D: Data<T>>: Clusterer<T, D> {
    /// Get cluster centroids
    ///
    /// # Returns
    /// Centroid coordinates for each cluster
    fn cluster_centers(&self) -> Option<Vec<Vec<T>>>;

    /// Get inertia (sum of squared distances to nearest cluster center)
    fn inertia(&self) -> Option<T> {
        None
    }

    /// Get number of iterations performed
    fn n_iterations(&self) -> usize {
        0
    }
}

/// Trait for density-based clustering algorithms (DBSCAN, etc.)
pub trait DensityClusterer<T: Numeric, D: Data<T>>: Clusterer<T, D> {
    /// Get core sample indices
    ///
    /// # Returns
    /// Indices of core samples
    fn core_sample_indices(&self) -> Option<Vec<usize>> {
        None
    }

    /// Check if a sample is a core sample
    fn is_core_sample(&self, index: usize) -> bool;

    /// Get noise samples (points not assigned to any cluster)
    ///
    /// # Returns
    /// Indices of noise samples
    fn noise_samples(&self) -> Vec<usize> {
        vec![]
    }
}

/// Trait for hierarchical clustering
pub trait HierarchicalClusterer<T: Numeric, D: Data<T>>: Clusterer<T, D> {
    /// Get the dendrogram/linkage matrix
    fn linkage(&self) -> Option<Vec<Vec<T>>> {
        None
    }

    /// Cut the dendrogram at a specific height
    fn cut_tree(&self, height: T) -> Result<Vec<usize>, String>;
}
