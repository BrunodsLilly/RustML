//! K-means clustering algorithm
//!
//! K-means is a centroid-based clustering algorithm that partitions data into k clusters
//! by iteratively assigning points to the nearest centroid and updating centroids.

use linear_algebra::matrix::Matrix;
use ml_traits::clustering::{CentroidClusterer, Clusterer};
use ml_traits::Data;
use rand::prelude::*;

/// Helper extension trait for Matrix to get rows as vectors (legacy)
///
/// NOTE: This allocates a new Vec on every call. For performance-critical code,
/// prefer using Matrix::row_slice() directly which returns a zero-copy slice.
trait MatrixExt {
    fn get_row(&self, row: usize) -> Vec<f64>;
}

impl MatrixExt for Matrix<f64> {
    fn get_row(&self, row: usize) -> Vec<f64> {
        self.row(row).unwrap().data
    }
}

/// K-means clustering algorithm
///
/// # Algorithm
/// 1. Initialize k centroids (random or k-means++)
/// 2. Assign each point to the nearest centroid
/// 3. Update centroids to the mean of assigned points
/// 4. Repeat until convergence or max iterations
///
/// # Example
/// ```
/// use clustering::KMeans;
/// use linear_algebra::matrix::Matrix;
/// use ml_traits::clustering::Clusterer;
///
/// let data = Matrix::from_vec(
///     vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 1.0, 0.6, 9.0, 11.0],
///     6,
///     2
/// ).unwrap();
///
/// let mut kmeans = KMeans::new(2, 100, 1e-4, Some(42));
/// kmeans.fit(&data).unwrap();
/// let labels = kmeans.predict(&data).unwrap();
/// assert_eq!(labels.len(), 6);
/// ```
pub struct KMeans {
    /// Number of clusters
    n_clusters: usize,
    /// Maximum number of iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Random seed for reproducibility
    seed: Option<u64>,
    /// Cluster centroids (k × n_features)
    centroids: Option<Vec<Vec<f64>>>,
    /// Inertia (sum of squared distances to nearest cluster center)
    inertia_value: Option<f64>,
    /// Number of iterations performed
    n_iterations: usize,
    /// Whether the model has been fitted
    fitted: bool,
}

impl KMeans {
    /// Create a new K-means clusterer
    ///
    /// # Arguments
    /// * `n_clusters` - Number of clusters to find
    /// * `max_iterations` - Maximum iterations before stopping
    /// * `tolerance` - Minimum centroid movement to continue
    /// * `seed` - Optional random seed for reproducibility
    pub fn new(
        n_clusters: usize,
        max_iterations: usize,
        tolerance: f64,
        seed: Option<u64>,
    ) -> Self {
        assert!(n_clusters > 0, "Number of clusters must be positive");
        assert!(max_iterations > 0, "Max iterations must be positive");
        assert!(
            tolerance > 0.0 && tolerance.is_finite(),
            "Tolerance must be positive and finite"
        );

        Self {
            n_clusters,
            max_iterations,
            tolerance,
            seed,
            centroids: None,
            inertia_value: None,
            n_iterations: 0,
            fitted: false,
        }
    }

    /// Initialize centroids using k-means++ algorithm for better convergence (optimized)
    fn initialize_centroids(&self, data: &Matrix<f64>) -> Vec<Vec<f64>> {
        let n_samples = data.n_samples();
        let n_features = data.n_features();
        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        let mut centroids = Vec::with_capacity(self.n_clusters);

        // Choose first centroid randomly (use row_slice for zero-copy)
        let first_idx = rng.gen_range(0..n_samples);
        centroids.push(data.row_slice(first_idx).expect("Valid row").to_vec());

        // Choose remaining centroids using k-means++
        for _ in 1..self.n_clusters {
            let mut distances = vec![f64::INFINITY; n_samples];

            // Calculate minimum distance to any existing centroid (zero-copy reads)
            for i in 0..n_samples {
                let point = data.row_slice(i).expect("Valid row");
                for centroid in &centroids {
                    let dist = Self::euclidean_distance(point, centroid);
                    distances[i] = distances[i].min(dist);
                }
            }

            // Square the distances for probability weighting
            let squared_distances: Vec<f64> = distances.iter().map(|d| d * d).collect();
            let sum: f64 = squared_distances.iter().sum();

            // Choose next centroid with probability proportional to squared distance
            let mut cumsum = 0.0;
            let threshold = rng.gen::<f64>() * sum;

            for (i, &sq_dist) in squared_distances.iter().enumerate() {
                cumsum += sq_dist;
                if cumsum >= threshold {
                    centroids.push(data.row_slice(i).expect("Valid row").to_vec());
                    break;
                }
            }
        }

        centroids
    }

    /// Assign each point to the nearest centroid (optimized with zero-copy)
    fn assign_clusters(&self, data: &Matrix<f64>, centroids: &[Vec<f64>]) -> Vec<usize> {
        let n_samples = data.n_samples();
        let mut labels = vec![0; n_samples];

        for i in 0..n_samples {
            // Use row_slice() for zero-copy access (10-20x faster than get_row())
            let point = data.row_slice(i).expect("Valid row index");
            let mut min_dist = f64::INFINITY;
            let mut best_cluster = 0;

            for (cluster_idx, centroid) in centroids.iter().enumerate() {
                let dist = Self::euclidean_distance(point, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = cluster_idx;
                }
            }

            labels[i] = best_cluster;
        }

        labels
    }

    /// Update centroids to the mean of assigned points (optimized with direct access)
    fn update_centroids(&self, data: &Matrix<f64>, labels: &[usize]) -> Vec<Vec<f64>> {
        let n_samples = data.n_samples();
        let n_features = data.n_features();
        let mut new_centroids = vec![vec![0.0; n_features]; self.n_clusters];
        let mut counts = vec![0; self.n_clusters];

        // Sum points in each cluster using direct matrix access (no row allocations)
        for i in 0..n_samples {
            let label = labels[i];
            counts[label] += 1;

            // Direct element access via get() - no row allocation
            for j in 0..n_features {
                let value = data.get(i, j).expect("Valid matrix index");
                new_centroids[label][j] += value;
            }
        }

        // Compute means
        for (cluster_idx, count) in counts.iter().enumerate() {
            if *count > 0 {
                for feature in &mut new_centroids[cluster_idx] {
                    *feature /= *count as f64;
                }
            }
        }

        new_centroids
    }

    /// Calculate Euclidean distance between two points
    fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Calculate maximum centroid movement
    fn max_centroid_shift(old: &[Vec<f64>], new: &[Vec<f64>]) -> f64 {
        old.iter()
            .zip(new.iter())
            .map(|(a, b)| Self::euclidean_distance(a, b))
            .fold(0.0, f64::max)
    }

    /// Calculate inertia (sum of squared distances to nearest cluster center)
    fn calculate_inertia(&self, data: &Matrix<f64>, labels: &[usize]) -> f64 {
        let centroids = self.centroids.as_ref().unwrap();
        let mut inertia = 0.0;

        for (i, &label) in labels.iter().enumerate() {
            let point = data.get_row(i);
            let dist = Self::euclidean_distance(&point, &centroids[label]);
            inertia += dist * dist;
        }

        inertia
    }
}

impl Clusterer<f64, Matrix<f64>> for KMeans {
    fn fit(&mut self, X: &Matrix<f64>) -> Result<(), String> {
        if X.n_samples() < self.n_clusters {
            return Err(format!(
                "Number of samples ({}) must be >= n_clusters ({})",
                X.n_samples(),
                self.n_clusters
            ));
        }

        // Initialize centroids using k-means++
        let mut centroids = self.initialize_centroids(X);

        // Main k-means loop
        for iteration in 0..self.max_iterations {
            // Assign points to nearest centroids
            let labels = self.assign_clusters(X, &centroids);

            // Update centroids
            let new_centroids = self.update_centroids(X, &labels);

            // Check convergence
            let shift = Self::max_centroid_shift(&centroids, &new_centroids);
            centroids = new_centroids;

            self.n_iterations = iteration + 1;

            if shift < self.tolerance {
                break;
            }
        }

        // Set centroids first, then calculate inertia
        self.centroids = Some(centroids);
        let labels = self.assign_clusters(X, &self.centroids.as_ref().unwrap());
        self.inertia_value = Some(self.calculate_inertia(X, &labels));
        self.fitted = true;

        Ok(())
    }

    fn predict(&self, X: &Matrix<f64>) -> Result<Vec<usize>, String> {
        if !self.is_fitted() {
            return Err("Model not fitted. Call fit() first.".to_string());
        }

        let centroids = self.centroids.as_ref().unwrap();
        Ok(self.assign_clusters(X, centroids))
    }

    fn fit_predict(&mut self, X: &Matrix<f64>) -> Result<Vec<usize>, String> {
        self.fit(X)?;
        self.predict(X)
    }

    fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl CentroidClusterer<f64, Matrix<f64>> for KMeans {
    fn cluster_centers(&self) -> Option<Vec<Vec<f64>>> {
        self.centroids.clone()
    }

    fn inertia(&self) -> Option<f64> {
        self.inertia_value
    }

    fn n_iterations(&self) -> usize {
        self.n_iterations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        // Two clear clusters
        let data = Matrix::from_vec(
            vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 1.0, 0.6, 9.0, 11.0],
            6,
            2,
        )
        .unwrap();

        let mut kmeans = KMeans::new(2, 100, 1e-4, Some(42));
        kmeans.fit(&data).unwrap();

        let labels = kmeans.predict(&data).unwrap();
        assert_eq!(labels.len(), 6);

        // Points in the same cluster should have the same label
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]); // Different clusters
    }

    #[test]
    fn test_kmeans_centroids() {
        let data = Matrix::from_vec(
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 10.0, 10.0, 10.0, 11.0, 11.0, 10.0,
            ],
            6,
            2,
        )
        .unwrap();

        let mut kmeans = KMeans::new(2, 100, 1e-4, Some(42));
        kmeans.fit(&data).unwrap();

        let centroids = kmeans.cluster_centers().unwrap();
        assert_eq!(centroids.len(), 2);
        assert_eq!(centroids[0].len(), 2);

        // One centroid should be near (0.33, 0.33), other near (10.33, 10.33)
        let mut has_low_centroid = false;
        let mut has_high_centroid = false;

        for centroid in centroids {
            if centroid[0] < 5.0 {
                has_low_centroid = true;
            } else {
                has_high_centroid = true;
            }
        }

        assert!(has_low_centroid && has_high_centroid);
    }

    #[test]
    fn test_kmeans_inertia() {
        let data = Matrix::from_vec(vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0], 3, 2).unwrap();

        let mut kmeans = KMeans::new(1, 100, 1e-4, Some(42));
        kmeans.fit(&data).unwrap();

        let inertia = kmeans.inertia().unwrap();
        assert!(inertia > 0.0);
        assert!(inertia.is_finite());
    }

    #[test]
    fn test_kmeans_not_fitted() {
        let kmeans = KMeans::new(2, 100, 1e-4, None);
        let data = Matrix::from_vec(vec![1.0, 2.0], 1, 2).unwrap();

        let result = kmeans.predict(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_kmeans_too_few_samples() {
        let data = Matrix::from_vec(vec![1.0, 2.0], 1, 2).unwrap();

        let mut kmeans = KMeans::new(2, 100, 1e-4, None);
        let result = kmeans.fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_kmeans_convergence() {
        // Data that should converge quickly
        let data =
            Matrix::from_vec(vec![0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1], 4, 2).unwrap();

        let mut kmeans = KMeans::new(2, 100, 1e-4, Some(42));
        kmeans.fit(&data).unwrap();

        // Should converge in few iterations for well-separated clusters
        assert!(kmeans.n_iterations() < 10);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 8.0];

        let dist = KMeans::euclidean_distance(&a, &b);
        let expected = ((3.0_f64).powi(2) + (4.0_f64).powi(2) + (5.0_f64).powi(2)).sqrt();
        assert!((dist - expected).abs() < 1e-10);
    }
}
