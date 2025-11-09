//! Statistical operations for data analysis and ML visualizations
//!
//! This module provides efficient implementations of correlation coefficients,
//! covariance, and other statistical measures needed for feature analysis.

use crate::matrix::Matrix;

/// Compute the mean of a vector
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Compute the variance of a vector
pub fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }

    let m = mean(data);
    let sum_sq_diff: f64 = data.iter().map(|&x| (x - m).powi(2)).sum();
    sum_sq_diff / (data.len() - 1) as f64
}

/// Compute the standard deviation of a vector
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Compute Pearson correlation coefficient between two vectors
///
/// Returns correlation in range [-1, 1]:
/// - 1: perfect positive correlation
/// - 0: no correlation
/// - -1: perfect negative correlation
///
/// # Arguments
/// * `x` - First variable
/// * `y` - Second variable
///
/// # Panics
/// Panics if vectors have different lengths
///
/// # Returns
/// Correlation coefficient, or 0.0 if either vector has zero variance
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Vectors must have same length");

    let n = x.len();
    if n < 2 {
        return 0.0;
    }

    let mean_x = mean(x);
    let mean_y = mean(y);

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    // Avoid division by zero
    if var_x < 1e-10 || var_y < 1e-10 {
        return 0.0;
    }

    cov / (var_x * var_y).sqrt()
}

/// Compute correlation matrix for all pairs of columns in a matrix
///
/// This is a vectorized, efficient implementation that computes all pairwise
/// correlations in a single pass through the data.
///
/// # Arguments
/// * `X` - Feature matrix where each column is a feature (n_samples × n_features)
///
/// # Returns
/// Symmetric correlation matrix (n_features × n_features) where:
/// - Diagonal elements are 1.0 (feature correlates perfectly with itself)
/// - Off-diagonal element (i,j) is correlation between features i and j
/// - Matrix is symmetric: corr(i,j) == corr(j,i)
///
/// # Example
/// ```
/// use linear_algebra::matrix::Matrix;
/// use linear_algebra::statistics::correlation_matrix;
///
/// // 3 samples, 2 features
/// let data = Matrix::from_vec(vec![
///     1.0, 2.0,
///     2.0, 4.0,
///     3.0, 6.0,
/// ], 3, 2).unwrap();
///
/// let corr = correlation_matrix(&data);
/// assert_eq!(corr.rows, 2);
/// assert_eq!(corr.cols, 2);
/// assert!((corr[(0, 0)] - 1.0).abs() < 1e-10); // Diagonal is 1.0
/// assert!((corr[(0, 1)] - 1.0).abs() < 1e-10); // Perfect correlation
/// ```
pub fn correlation_matrix(X: &Matrix<f64>) -> Matrix<f64> {
    let n_features = X.cols;
    let n_samples = X.rows as f64;

    if X.rows < 2 {
        return Matrix::identity(n_features);
    }

    // Step 1: Compute column means
    let mut means = vec![0.0; n_features];
    for j in 0..n_features {
        let col = X.col(j).expect("Column index valid");
        means[j] = mean(&col.data);
    }

    // Step 2: Center the data (subtract column means)
    let mut centered_data = vec![0.0; X.rows * X.cols];
    for i in 0..X.rows {
        for j in 0..X.cols {
            centered_data[i * X.cols + j] = X[(i, j)] - means[j];
        }
    }
    let centered =
        Matrix::from_vec(centered_data, X.rows, X.cols).expect("Same dimensions as input");

    // Step 3: Compute covariance matrix: (X^T * X) / (n-1)
    // This is efficient because we compute all pairwise covariances at once
    let mut cov = Matrix::zeros(n_features, n_features);
    for i in 0..n_features {
        for j in 0..n_features {
            let mut sum = 0.0;
            for k in 0..X.rows {
                sum += centered[(k, i)] * centered[(k, j)];
            }
            cov[(i, j)] = sum / (n_samples - 1.0);
        }
    }

    // Step 4: Compute standard deviations for each feature
    let mut std_devs = vec![0.0; n_features];
    for j in 0..n_features {
        std_devs[j] = cov[(j, j)].sqrt().max(1e-10); // Avoid division by zero
    }

    // Step 5: Normalize covariance to get correlation
    let mut corr = Matrix::zeros(n_features, n_features);
    for i in 0..n_features {
        for j in 0..n_features {
            corr[(i, j)] = cov[(i, j)] / (std_devs[i] * std_devs[j]);
        }
    }

    corr
}

/// Standardize features by removing mean and scaling to unit variance
///
/// This is useful for computing standardized regression coefficients
/// (coefficients that show relative importance regardless of feature scale)
///
/// # Arguments
/// * `X` - Feature matrix (n_samples × n_features)
///
/// # Returns
/// Tuple of (standardized_matrix, means, std_devs)
pub fn standardize(X: &Matrix<f64>) -> (Matrix<f64>, Vec<f64>, Vec<f64>) {
    let n_features = X.cols;

    // Compute means and standard deviations
    let mut means = vec![0.0; n_features];
    let mut std_devs = vec![0.0; n_features];

    for j in 0..n_features {
        let col = X.col(j).expect("Column index valid");
        means[j] = mean(&col.data);
        std_devs[j] = std_dev(&col.data).max(1e-10); // Avoid division by zero
    }

    // Standardize
    let mut standardized_data = vec![0.0; X.rows * X.cols];
    for i in 0..X.rows {
        for j in 0..X.cols {
            standardized_data[i * X.cols + j] = (X[(i, j)] - means[j]) / std_devs[j];
        }
    }

    let standardized =
        Matrix::from_vec(standardized_data, X.rows, X.cols).expect("Same dimensions as input");

    (standardized, means, std_devs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&data) - 3.0).abs() < 1e-10);

        let empty: Vec<f64> = vec![];
        assert_eq!(mean(&empty), 0.0);
    }

    #[test]
    fn test_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let var = variance(&data);
        assert!((var - 4.571428).abs() < 1e-5);
    }

    #[test]
    fn test_std_dev() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&data);
        assert!((sd - 2.138089).abs() < 1e-5);
    }

    #[test]
    fn test_pearson_correlation_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = pearson_correlation(&x, &y);
        assert!(
            (corr - 1.0).abs() < 1e-10,
            "Perfect positive correlation should be 1.0"
        );
    }

    #[test]
    fn test_pearson_correlation_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let corr = pearson_correlation(&x, &y);
        assert!(
            (corr + 1.0).abs() < 1e-10,
            "Perfect negative correlation should be -1.0"
        );
    }

    #[test]
    fn test_pearson_correlation_no_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 3.0, 7.0, 2.0, 6.0];
        let corr = pearson_correlation(&x, &y);
        // Not exactly 0, but close to it
        assert!(corr.abs() < 0.5, "Should have low correlation");
    }

    #[test]
    fn test_pearson_correlation_zero_variance() {
        let x = vec![5.0, 5.0, 5.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let corr = pearson_correlation(&x, &y);
        assert_eq!(corr, 0.0, "Zero variance should give 0 correlation");
    }

    #[test]
    fn test_correlation_matrix_identity() {
        // Single feature should give identity matrix
        let data = Matrix::from_vec(vec![1.0, 2.0, 3.0], 3, 1).unwrap();
        let corr = correlation_matrix(&data);
        assert_eq!(corr.rows, 1);
        assert_eq!(corr.cols, 1);
        assert!((corr[(0, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_matrix_perfect_correlation() {
        // Two perfectly correlated features
        let data = Matrix::from_vec(vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0], 3, 2).unwrap();

        let corr = correlation_matrix(&data);
        assert_eq!(corr.rows, 2);
        assert_eq!(corr.cols, 2);

        // Diagonal should be 1.0
        assert!((corr[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((corr[(1, 1)] - 1.0).abs() < 1e-10);

        // Off-diagonal should be 1.0 (perfect correlation)
        assert!((corr[(0, 1)] - 1.0).abs() < 1e-10);
        assert!((corr[(1, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_matrix_symmetric() {
        let data = Matrix::from_vec(
            vec![1.0, 2.0, 3.0, 2.0, 3.0, 5.0, 3.0, 4.0, 7.0, 4.0, 5.0, 9.0],
            4,
            3,
        )
        .unwrap();

        let corr = correlation_matrix(&data);

        // Matrix should be symmetric
        for i in 0..corr.rows {
            for j in 0..corr.cols {
                assert!(
                    (corr[(i, j)] - corr[(j, i)]).abs() < 1e-10,
                    "Correlation matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_standardize() {
        let data = Matrix::from_vec(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0], 3, 2).unwrap();

        let (standardized, means, _std_devs) = standardize(&data);

        // Check means
        assert!((means[0] - 2.0).abs() < 1e-10);
        assert!((means[1] - 20.0).abs() < 1e-10);

        // Check that standardized data has mean ~0
        for j in 0..standardized.cols {
            let col = standardized.col(j).unwrap();
            let col_mean = mean(&col.data);
            assert!(
                col_mean.abs() < 1e-10,
                "Standardized columns should have mean 0"
            );
        }

        // Check that standardized data has std ~1
        for j in 0..standardized.cols {
            let col = standardized.col(j).unwrap();
            let col_std = std_dev(&col.data);
            assert!(
                (col_std - 1.0).abs() < 1e-10,
                "Standardized columns should have std 1"
            );
        }
    }
}
