//! Principal Component Analysis (PCA)
//!
//! PCA is a linear dimensionality reduction technique that finds orthogonal axes
//! (principal components) that capture the maximum variance in the data.

use linear_algebra::matrix::Matrix;
use linear_algebra::statistics::{correlation_matrix, standardize};
use ml_traits::reduction::LinearReduction;
use ml_traits::unsupervised::{DimensionalityReduction, UnsupervisedModel};
use ml_traits::Data;

/// Principal Component Analysis
///
/// # Algorithm
/// 1. Standardize the data (zero mean, unit variance)
/// 2. Compute the covariance or correlation matrix
/// 3. Compute eigenvalues and eigenvectors
/// 4. Sort by eigenvalue magnitude (descending)
/// 5. Select top k eigenvectors as principal components
/// 6. Transform data by projection onto principal components
///
/// # Example
/// ```
/// use dimensionality_reduction::PCA;
/// use linear_algebra::matrix::Matrix;
/// use ml_traits::unsupervised::UnsupervisedModel;
///
/// // 4 samples, 3 features
/// let data = Matrix::from_vec(
///     vec![
///         2.5, 2.4,
///         0.5, 0.7,
///         2.2, 2.9,
///         1.9, 2.2,
///         3.1, 3.0,
///         2.3, 2.7,
///     ],
///     6,
///     2,
/// ).unwrap();
///
/// let mut pca = PCA::new(1); // Reduce to 1 dimension
/// pca.fit(&data).unwrap();
/// let transformed = pca.transform(&data).unwrap();
/// assert_eq!(transformed.len(), 6); // 6 samples
/// assert_eq!(transformed[0].len(), 1); // 1 component
/// ```
pub struct PCA {
    /// Number of components to keep
    n_components: usize,
    /// Principal components (eigenvectors)
    components: Option<Vec<Vec<f64>>>,
    /// Explained variance (eigenvalues)
    explained_variance: Option<Vec<f64>>,
    /// Mean of each feature (for centering)
    mean: Option<Vec<f64>>,
    /// Standard deviation of each feature (for scaling)
    std: Option<Vec<f64>>,
    /// Whether the model has been fitted
    fitted: bool,
}

impl PCA {
    /// Create a new PCA instance
    ///
    /// # Arguments
    /// * `n_components` - Number of principal components to keep
    pub fn new(n_components: usize) -> Self {
        assert!(n_components > 0, "Number of components must be positive");

        Self {
            n_components,
            components: None,
            explained_variance: None,
            mean: None,
            std: None,
            fitted: false,
        }
    }

    /// Get the principal components (eigenvectors)
    pub fn components(&self) -> Option<&Vec<Vec<f64>>> {
        self.components.as_ref()
    }

    /// Get the explained variance (eigenvalues)
    pub fn explained_variance(&self) -> Option<&Vec<f64>> {
        self.explained_variance.as_ref()
    }

    /// Compute eigenvalues and eigenvectors using power iteration
    /// This is a simple implementation; production code would use LAPACK
    fn power_iteration(matrix: &Matrix<f64>, n_iter: usize) -> (f64, Vec<f64>) {
        let n = matrix.rows;
        let mut v = vec![1.0 / (n as f64).sqrt(); n];

        for _ in 0..n_iter {
            // Matrix-vector multiplication
            let mut new_v = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    new_v[i] += matrix.data[i * matrix.cols + j] * v[j];
                }
            }

            // Normalize
            let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v = new_v.iter().map(|x| x / norm).collect();
        }

        // Compute eigenvalue: λ = v^T * A * v
        let mut av = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                av[i] += matrix.data[i * matrix.cols + j] * v[j];
            }
        }
        let eigenvalue: f64 = v.iter().zip(av.iter()).map(|(a, b)| a * b).sum();

        (eigenvalue, v)
    }

    /// Deflation: Remove the influence of an eigenvector from the matrix
    fn deflate(matrix: &mut Matrix<f64>, eigenvalue: f64, eigenvector: &[f64]) {
        let n = matrix.rows;
        for i in 0..n {
            for j in 0..n {
                let idx = i * matrix.cols + j;
                matrix.data[idx] -= eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }
    }

    /// Compute top k eigenvalues and eigenvectors
    fn compute_eigen(matrix: &Matrix<f64>, k: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut work_matrix = matrix.clone();
        let mut eigenvalues = Vec::with_capacity(k);
        let mut eigenvectors = Vec::with_capacity(k);

        for _ in 0..k {
            let (eigenvalue, eigenvector) = Self::power_iteration(&work_matrix, 100);

            // Only keep if eigenvalue is significant
            if eigenvalue.abs() > 1e-10 {
                eigenvalues.push(eigenvalue);
                eigenvectors.push(eigenvector.clone());
                Self::deflate(&mut work_matrix, eigenvalue, &eigenvector);
            } else {
                break;
            }
        }

        (eigenvalues, eigenvectors)
    }
}

impl UnsupervisedModel<f64, Matrix<f64>> for PCA {
    fn fit(&mut self, X: &Matrix<f64>) -> Result<(), String> {
        let n_samples = X.n_samples();
        let n_features = X.n_features();

        if self.n_components > n_features {
            return Err(format!(
                "n_components ({}) cannot exceed n_features ({})",
                self.n_components, n_features
            ));
        }

        if n_samples < 2 {
            return Err("Need at least 2 samples for PCA".to_string());
        }

        // Standardize the data
        let (standardized, mean, std) = standardize(X);

        // Compute correlation matrix (equivalent to covariance for standardized data)
        let corr_matrix = correlation_matrix(&standardized);

        // Compute eigenvalues and eigenvectors
        let (eigenvalues, eigenvectors) = Self::compute_eigen(&corr_matrix, self.n_components);

        if eigenvalues.len() < self.n_components {
            return Err(format!(
                "Could only find {} significant components (requested {})",
                eigenvalues.len(),
                self.n_components
            ));
        }

        self.components = Some(eigenvectors);
        self.explained_variance = Some(eigenvalues);
        self.mean = Some(mean);
        self.std = Some(std);
        self.fitted = true;

        Ok(())
    }

    fn transform(&self, X: &Matrix<f64>) -> Result<Vec<Vec<f64>>, String> {
        if !self.is_fitted() {
            return Err("Model not fitted. Call fit() first.".to_string());
        }

        let n_samples = X.n_samples();
        let n_features = X.n_features();
        let components = self.components.as_ref().unwrap();
        let mean = self.mean.as_ref().unwrap();
        let std = self.std.as_ref().unwrap();

        if n_features != mean.len() {
            return Err(format!(
                "X has {} features, but PCA was fitted with {} features",
                n_features,
                mean.len()
            ));
        }

        let mut transformed = vec![vec![0.0; self.n_components]; n_samples];

        // For each sample
        for i in 0..n_samples {
            // Standardize the sample
            let mut standardized = vec![0.0; n_features];
            for j in 0..n_features {
                let val = X.get(i, j).unwrap();
                standardized[j] = if std[j] > 1e-10 {
                    (val - mean[j]) / std[j]
                } else {
                    val - mean[j]
                };
            }

            // Project onto each principal component
            for (comp_idx, component) in components.iter().enumerate() {
                let mut projection = 0.0;
                for (feat_idx, &feat_val) in standardized.iter().enumerate() {
                    projection += feat_val * component[feat_idx];
                }
                transformed[i][comp_idx] = projection;
            }
        }

        Ok(transformed)
    }

    fn fit_transform(&mut self, X: &Matrix<f64>) -> Result<Vec<Vec<f64>>, String> {
        self.fit(X)?;
        self.transform(X)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl DimensionalityReduction<f64, Matrix<f64>> for PCA {
    fn n_components(&self) -> usize {
        self.n_components
    }

    fn explained_variance_ratio(&self) -> Option<Vec<f64>> {
        self.explained_variance.as_ref().map(|var| {
            let total: f64 = var.iter().sum();
            var.iter().map(|v| v / total).collect()
        })
    }

    fn components(&self) -> Option<Vec<Vec<f64>>> {
        self.components.clone()
    }
}

impl LinearReduction<f64, Matrix<f64>> for PCA {
    fn singular_values(&self) -> Option<Vec<f64>> {
        // For PCA, singular values are sqrt of eigenvalues
        self.explained_variance
            .as_ref()
            .map(|eig| eig.iter().map(|e| e.sqrt()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_basic() {
        // Simple 2D data that should reduce to 1D
        let data = Matrix::from_vec(
            vec![
                1.0, 1.0,
                2.0, 2.0,
                3.0, 3.0,
                4.0, 4.0,
            ],
            4,
            2,
        )
        .unwrap();

        let mut pca = PCA::new(1);
        let result = pca.fit_transform(&data);

        assert!(result.is_ok());
        let transformed = result.unwrap();
        assert_eq!(transformed.len(), 4);
        assert_eq!(transformed[0].len(), 1);
    }

    #[test]
    fn test_pca_explained_variance() {
        let data = Matrix::from_vec(
            vec![
                2.5, 2.4,
                0.5, 0.7,
                2.2, 2.9,
                1.9, 2.2,
                3.1, 3.0,
                2.3, 2.7,
            ],
            6,
            2,
        )
        .unwrap();

        let mut pca = PCA::new(2);
        pca.fit(&data).unwrap();

        let variance_ratio = pca.explained_variance_ratio().unwrap();
        assert_eq!(variance_ratio.len(), 2);

        // Sum of variance ratios should be ~1.0
        let sum: f64 = variance_ratio.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        // First component should explain more variance
        assert!(variance_ratio[0] > variance_ratio[1]);
    }

    #[test]
    fn test_pca_components() {
        let data = Matrix::from_vec(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], 3, 2).unwrap();

        let mut pca = PCA::new(1);
        pca.fit(&data).unwrap();

        let components = pca.components().unwrap();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 2);

        // Component should be unit vector
        let norm: f64 = components[0].iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pca_not_fitted() {
        let pca = PCA::new(1);
        let data = Matrix::from_vec(vec![1.0, 2.0], 1, 2).unwrap();

        let result = pca.transform(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_pca_too_many_components() {
        let data = Matrix::from_vec(vec![1.0, 2.0], 1, 2).unwrap();

        let mut pca = PCA::new(3); // More than 2 features
        let result = pca.fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_pca_inverse_transform_shape() {
        let data = Matrix::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            3,
            2,
        )
        .unwrap();

        let mut pca = PCA::new(1);
        let transformed = pca.fit_transform(&data).unwrap();

        assert_eq!(transformed.len(), 3); // 3 samples
        assert_eq!(transformed[0].len(), 1); // 1 component
    }
}
