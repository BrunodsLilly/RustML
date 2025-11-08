//! Feature scaling transformers
//!
//! Scalers transform features to a common scale without distorting differences in ranges.

use linear_algebra::matrix::Matrix;
use ml_traits::preprocessing::{Scaler, Transformer};
use ml_traits::Data as DataTrait;

/// Helper trait for Matrix row access
trait MatrixExt {
    fn get_row(&self, row: usize) -> Vec<f64>;
}

impl MatrixExt for Matrix<f64> {
    fn get_row(&self, row: usize) -> Vec<f64> {
        self.row(row).unwrap().data
    }
}

/// Standard Scaler (Z-score normalization)
///
/// Standardizes features by removing the mean and scaling to unit variance:
/// z = (x - μ) / σ
///
/// # Example
/// ```
/// use preprocessing::StandardScaler;
/// use linear_algebra::matrix::Matrix;
/// use ml_traits::preprocessing::Transformer;
///
/// let data = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
///
/// let mut scaler = StandardScaler::new();
/// scaler.fit(&data).unwrap();
/// let scaled = scaler.transform(&data).unwrap();
///
/// assert_eq!(scaled.len(), 3); // 3 samples
/// assert_eq!(scaled[0].len(), 2); // 2 features
/// ```
#[derive(Debug, Clone)]
pub struct StandardScaler {
    /// Mean of each feature
    mean: Option<Vec<f64>>,
    /// Standard deviation of each feature
    std: Option<Vec<f64>>,
    /// Whether the scaler has been fitted
    fitted: bool,
}

impl StandardScaler {
    /// Create a new StandardScaler
    pub fn new() -> Self {
        Self {
            mean: None,
            std: None,
            fitted: false,
        }
    }

    /// Get the mean values
    pub fn mean(&self) -> Option<&Vec<f64>> {
        self.mean.as_ref()
    }

    /// Get the standard deviation values
    pub fn std(&self) -> Option<&Vec<f64>> {
        self.std.as_ref()
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer<f64, Matrix<f64>> for StandardScaler {
    fn fit(&mut self, X: &Matrix<f64>) -> Result<(), String> {
        let n_samples = X.n_samples();
        let n_features = X.n_features();

        if n_samples == 0 {
            return Err("Cannot fit on empty data".to_string());
        }

        // Calculate mean for each feature
        let mut mean = vec![0.0; n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                mean[j] += DataTrait::get(X, i, j).unwrap();
            }
        }
        for m in &mut mean {
            *m /= n_samples as f64;
        }

        // Calculate standard deviation for each feature
        let mut variance = vec![0.0; n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                let diff = DataTrait::get(X, i, j).unwrap() - mean[j];
                variance[j] += diff * diff;
            }
        }

        let std: Vec<f64> = variance
            .iter()
            .map(|v| {
                let std = (v / n_samples as f64).sqrt();
                if std < 1e-10 {
                    1.0 // Avoid division by zero
                } else {
                    std
                }
            })
            .collect();

        self.mean = Some(mean);
        self.std = Some(std);
        self.fitted = true;

        Ok(())
    }

    fn transform(&self, X: &Matrix<f64>) -> Result<Vec<Vec<f64>>, String> {
        if !self.is_fitted() {
            return Err("Scaler not fitted. Call fit() first.".to_string());
        }

        let n_samples = X.n_samples();
        let n_features = X.n_features();
        let mean = self.mean.as_ref().unwrap();
        let std = self.std.as_ref().unwrap();

        if n_features != mean.len() {
            return Err(format!(
                "X has {} features, but scaler was fitted with {} features",
                n_features,
                mean.len()
            ));
        }

        let mut transformed = vec![vec![0.0; n_features]; n_samples];

        for i in 0..n_samples {
            for j in 0..n_features {
                let val = DataTrait::get(X, i, j).unwrap();
                transformed[i][j] = (val - mean[j]) / std[j];
            }
        }

        Ok(transformed)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl Scaler<f64, Matrix<f64>> for StandardScaler {
    fn scale(&self) -> Option<Vec<f64>> {
        self.std.clone()
    }

    fn offset(&self) -> Option<Vec<f64>> {
        self.mean.clone()
    }

    fn inverse_transform(&self, X: &Matrix<f64>) -> Result<Vec<Vec<f64>>, String> {
        if !self.is_fitted() {
            return Err("Scaler not fitted. Call fit() first.".to_string());
        }

        let n_samples = X.n_samples();
        let n_features = X.n_features();
        let mean = self.mean.as_ref().unwrap();
        let std = self.std.as_ref().unwrap();

        if n_features != mean.len() {
            return Err(format!(
                "X has {} features, but scaler was fitted with {} features",
                n_features,
                mean.len()
            ));
        }

        let mut transformed = vec![vec![0.0; n_features]; n_samples];

        for i in 0..n_samples {
            for j in 0..n_features {
                let val = DataTrait::get(X, i, j).unwrap();
                transformed[i][j] = val * std[j] + mean[j];
            }
        }

        Ok(transformed)
    }
}

/// MinMax Scaler
///
/// Scales features to a given range [min, max] (default [0, 1]):
/// x_scaled = (x - x_min) / (x_max - x_min) * (max - min) + min
///
/// # Example
/// ```
/// use preprocessing::MinMaxScaler;
/// use linear_algebra::matrix::Matrix;
/// use ml_traits::preprocessing::Transformer;
///
/// let data = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
///
/// let mut scaler = MinMaxScaler::new(0.0, 1.0);
/// scaler.fit(&data).unwrap();
/// let scaled = scaler.transform(&data).unwrap();
///
/// // Values should be in [0, 1] range
/// for row in &scaled {
///     for &val in row {
///         assert!(val >= 0.0 && val <= 1.0);
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    /// Minimum value of target range
    feature_range_min: f64,
    /// Maximum value of target range
    feature_range_max: f64,
    /// Minimum value of each feature in training data
    data_min: Option<Vec<f64>>,
    /// Maximum value of each feature in training data
    data_max: Option<Vec<f64>>,
    /// Whether the scaler has been fitted
    fitted: bool,
}

impl MinMaxScaler {
    /// Create a new MinMaxScaler
    ///
    /// # Arguments
    /// * `min` - Minimum value of target range
    /// * `max` - Maximum value of target range
    pub fn new(min: f64, max: f64) -> Self {
        assert!(min < max, "min must be less than max");

        Self {
            feature_range_min: min,
            feature_range_max: max,
            data_min: None,
            data_max: None,
            fitted: false,
        }
    }

    /// Get the data minimum values
    pub fn data_min(&self) -> Option<&Vec<f64>> {
        self.data_min.as_ref()
    }

    /// Get the data maximum values
    pub fn data_max(&self) -> Option<&Vec<f64>> {
        self.data_max.as_ref()
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new(0.0, 1.0)
    }
}

impl Transformer<f64, Matrix<f64>> for MinMaxScaler {
    fn fit(&mut self, X: &Matrix<f64>) -> Result<(), String> {
        let n_samples = X.n_samples();
        let n_features = X.n_features();

        if n_samples == 0 {
            return Err("Cannot fit on empty data".to_string());
        }

        // Find min and max for each feature
        let mut data_min = vec![f64::INFINITY; n_features];
        let mut data_max = vec![f64::NEG_INFINITY; n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                let val = DataTrait::get(X, i, j).unwrap();
                data_min[j] = data_min[j].min(val);
                data_max[j] = data_max[j].max(val);
            }
        }

        self.data_min = Some(data_min);
        self.data_max = Some(data_max);
        self.fitted = true;

        Ok(())
    }

    fn transform(&self, X: &Matrix<f64>) -> Result<Vec<Vec<f64>>, String> {
        if !self.is_fitted() {
            return Err("Scaler not fitted. Call fit() first.".to_string());
        }

        let n_samples = X.n_samples();
        let n_features = X.n_features();
        let data_min = self.data_min.as_ref().unwrap();
        let data_max = self.data_max.as_ref().unwrap();

        if n_features != data_min.len() {
            return Err(format!(
                "X has {} features, but scaler was fitted with {} features",
                n_features,
                data_min.len()
            ));
        }

        let mut transformed = vec![vec![0.0; n_features]; n_samples];
        let range = self.feature_range_max - self.feature_range_min;

        for i in 0..n_samples {
            for j in 0..n_features {
                let val = DataTrait::get(X, i, j).unwrap();
                let data_range = data_max[j] - data_min[j];

                transformed[i][j] = if data_range < 1e-10 {
                    self.feature_range_min // Constant feature
                } else {
                    ((val - data_min[j]) / data_range) * range + self.feature_range_min
                };
            }
        }

        Ok(transformed)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl Scaler<f64, Matrix<f64>> for MinMaxScaler {
    fn scale(&self) -> Option<Vec<f64>> {
        if let (Some(min), Some(max)) = (&self.data_min, &self.data_max) {
            Some(
                min.iter()
                    .zip(max.iter())
                    .map(|(mi, ma)| ma - mi)
                    .collect(),
            )
        } else {
            None
        }
    }

    fn offset(&self) -> Option<Vec<f64>> {
        self.data_min.clone()
    }

    fn inverse_transform(&self, X: &Matrix<f64>) -> Result<Vec<Vec<f64>>, String> {
        if !self.is_fitted() {
            return Err("Scaler not fitted. Call fit() first.".to_string());
        }

        let n_samples = X.n_samples();
        let n_features = X.n_features();
        let data_min = self.data_min.as_ref().unwrap();
        let data_max = self.data_max.as_ref().unwrap();

        if n_features != data_min.len() {
            return Err(format!(
                "X has {} features, but scaler was fitted with {} features",
                n_features,
                data_min.len()
            ));
        }

        let mut transformed = vec![vec![0.0; n_features]; n_samples];
        let range = self.feature_range_max - self.feature_range_min;

        for i in 0..n_samples {
            for j in 0..n_features {
                let val = DataTrait::get(X, i, j).unwrap();
                let data_range = data_max[j] - data_min[j];

                transformed[i][j] = if data_range < 1e-10 {
                    data_min[j] // Constant feature
                } else {
                    ((val - self.feature_range_min) / range) * data_range + data_min[j]
                };
            }
        }

        Ok(transformed)
    }
}

/// Robust Scaler
///
/// Scales features using statistics that are robust to outliers:
/// x_scaled = (x - median) / IQR
///
/// Uses median and interquartile range instead of mean and standard deviation.
///
/// # Example
/// ```
/// use preprocessing::RobustScaler;
/// use linear_algebra::matrix::Matrix;
/// use ml_traits::preprocessing::Transformer;
///
/// let data = Matrix::from_vec(vec![1.0, 2.0, 3.0, 100.0, 5.0, 6.0], 3, 2).unwrap();
///
/// let mut scaler = RobustScaler::new();
/// scaler.fit(&data).unwrap();
/// let scaled = scaler.transform(&data).unwrap();
///
/// assert_eq!(scaled.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct RobustScaler {
    /// Median of each feature
    median: Option<Vec<f64>>,
    /// Interquartile range (IQR) of each feature
    iqr: Option<Vec<f64>>,
    /// Whether the scaler has been fitted
    fitted: bool,
}

impl RobustScaler {
    /// Create a new RobustScaler
    pub fn new() -> Self {
        Self {
            median: None,
            iqr: None,
            fitted: false,
        }
    }

    /// Calculate median of a sorted vector
    fn median(sorted: &[f64]) -> f64 {
        let n = sorted.len();
        if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    }

    /// Calculate percentile of a sorted vector
    fn percentile(sorted: &[f64], p: f64) -> f64 {
        let n = sorted.len();
        let idx = (p * (n - 1) as f64).floor() as usize;
        let frac = p * (n - 1) as f64 - idx as f64;

        if idx + 1 < n {
            sorted[idx] * (1.0 - frac) + sorted[idx + 1] * frac
        } else {
            sorted[idx]
        }
    }
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer<f64, Matrix<f64>> for RobustScaler {
    fn fit(&mut self, X: &Matrix<f64>) -> Result<(), String> {
        let n_samples = X.n_samples();
        let n_features = X.n_features();

        if n_samples == 0 {
            return Err("Cannot fit on empty data".to_string());
        }

        let mut median = vec![0.0; n_features];
        let mut iqr = vec![0.0; n_features];

        // Calculate median and IQR for each feature
        for j in 0..n_features {
            // Collect feature values
            let mut values = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                values.push(DataTrait::get(X, i, j).unwrap());
            }

            // Sort for percentile calculations
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            median[j] = Self::median(&values);
            let q25 = Self::percentile(&values, 0.25);
            let q75 = Self::percentile(&values, 0.75);
            iqr[j] = q75 - q25;

            // Avoid division by zero
            if iqr[j] < 1e-10 {
                iqr[j] = 1.0;
            }
        }

        self.median = Some(median);
        self.iqr = Some(iqr);
        self.fitted = true;

        Ok(())
    }

    fn transform(&self, X: &Matrix<f64>) -> Result<Vec<Vec<f64>>, String> {
        if !self.is_fitted() {
            return Err("Scaler not fitted. Call fit() first.".to_string());
        }

        let n_samples = X.n_samples();
        let n_features = X.n_features();
        let median = self.median.as_ref().unwrap();
        let iqr = self.iqr.as_ref().unwrap();

        if n_features != median.len() {
            return Err(format!(
                "X has {} features, but scaler was fitted with {} features",
                n_features,
                median.len()
            ));
        }

        let mut transformed = vec![vec![0.0; n_features]; n_samples];

        for i in 0..n_samples {
            for j in 0..n_features {
                let val = DataTrait::get(X, i, j).unwrap();
                transformed[i][j] = (val - median[j]) / iqr[j];
            }
        }

        Ok(transformed)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl Scaler<f64, Matrix<f64>> for RobustScaler {
    fn scale(&self) -> Option<Vec<f64>> {
        self.iqr.clone()
    }

    fn offset(&self) -> Option<Vec<f64>> {
        self.median.clone()
    }

    fn inverse_transform(&self, X: &Matrix<f64>) -> Result<Vec<Vec<f64>>, String> {
        if !self.is_fitted() {
            return Err("Scaler not fitted. Call fit() first.".to_string());
        }

        let n_samples = X.n_samples();
        let n_features = X.n_features();
        let median = self.median.as_ref().unwrap();
        let iqr = self.iqr.as_ref().unwrap();

        if n_features != median.len() {
            return Err(format!(
                "X has {} features, but scaler was fitted with {} features",
                n_features,
                median.len()
            ));
        }

        let mut transformed = vec![vec![0.0; n_features]; n_samples];

        for i in 0..n_samples {
            for j in 0..n_features {
                let val = DataTrait::get(X, i, j).unwrap();
                transformed[i][j] = val * iqr[j] + median[j];
            }
        }

        Ok(transformed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_scaler() {
        let data = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();

        let mut scaler = StandardScaler::new();
        scaler.fit(&data).unwrap();

        let mean = scaler.mean().unwrap();
        assert_eq!(mean.len(), 2);
        assert!((mean[0] - 3.0).abs() < 1e-10);
        assert!((mean[1] - 4.0).abs() < 1e-10);

        let scaled = scaler.transform(&data).unwrap();
        assert_eq!(scaled.len(), 3);

        // Check that scaled data has zero mean
        let scaled_mean: f64 = scaled.iter().map(|row| row[0]).sum::<f64>() / 3.0;
        assert!(scaled_mean.abs() < 1e-10);
    }

    #[test]
    fn test_standard_scaler_inverse() {
        let data = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();

        let mut scaler = StandardScaler::new();
        scaler.fit(&data).unwrap();

        let scaled_data = Matrix::from_vec(
            scaler.transform(&data).unwrap().into_iter().flatten().collect(),
            3,
            2,
        )
        .unwrap();

        let inverse = scaler.inverse_transform(&scaled_data).unwrap();

        // Should recover original data
        for i in 0..3 {
            for j in 0..2 {
                let original = data.get(i, j).unwrap();
                assert!((inverse[i][j] - original).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_minmax_scaler() {
        let data = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();

        let mut scaler = MinMaxScaler::new(0.0, 1.0);
        scaler.fit(&data).unwrap();

        let scaled = scaler.transform(&data).unwrap();

        // All values should be in [0, 1]
        for row in &scaled {
            for &val in row {
                assert!(val >= -1e-10 && val <= 1.0 + 1e-10);
            }
        }

        // Min should map to 0, max to 1
        assert!(scaled[0][0].abs() < 1e-10); // min of column 0
        assert!((scaled[2][0] - 1.0).abs() < 1e-10); // max of column 0
    }

    #[test]
    fn test_minmax_scaler_inverse() {
        let data = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();

        let mut scaler = MinMaxScaler::new(0.0, 1.0);
        scaler.fit(&data).unwrap();

        let scaled_data = Matrix::from_vec(
            scaler.transform(&data).unwrap().into_iter().flatten().collect(),
            3,
            2,
        )
        .unwrap();

        let inverse = scaler.inverse_transform(&scaled_data).unwrap();

        // Should recover original data
        for i in 0..3 {
            for j in 0..2 {
                let original = data.get(i, j).unwrap();
                assert!((inverse[i][j] - original).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_robust_scaler() {
        let data = Matrix::from_vec(
            vec![1.0, 2.0, 3.0, 100.0, 5.0, 6.0], // 100 is an outlier
            3,
            2,
        )
        .unwrap();

        let mut scaler = RobustScaler::new();
        scaler.fit(&data).unwrap();

        let scaled = scaler.transform(&data).unwrap();
        assert_eq!(scaled.len(), 3);

        // Median should be robust to outliers
        let median = scaler.median.as_ref().unwrap();
        assert!((median[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_scaler_not_fitted() {
        let scaler = StandardScaler::new();
        let data = Matrix::from_vec(vec![1.0, 2.0], 1, 2).unwrap();

        let result = scaler.transform(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_minmax_custom_range() {
        let data = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();

        let mut scaler = MinMaxScaler::new(-1.0, 1.0);
        scaler.fit(&data).unwrap();

        let scaled = scaler.transform(&data).unwrap();

        // All values should be in [-1, 1]
        for row in &scaled {
            for &val in row {
                assert!(val >= -1.0 - 1e-10 && val <= 1.0 + 1e-10);
            }
        }
    }
}
