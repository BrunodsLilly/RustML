//! Traits for data preprocessing and transformation

use crate::{Data, Numeric};

/// Core trait for data transformers
///
/// Transformers modify data in some way (scaling, encoding, etc.)
pub trait Transformer<T: Numeric, D: Data<T>> {
    /// Fit the transformer to data
    ///
    /// # Arguments
    /// * `X` - Data to fit
    ///
    /// # Returns
    /// Result indicating success or error
    fn fit(&mut self, X: &D) -> Result<(), String>;

    /// Transform data using the fitted transformer
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

    /// Check if the transformer has been fitted
    fn is_fitted(&self) -> bool;
}

/// Trait for scalers (StandardScaler, MinMaxScaler, etc.)
pub trait Scaler<T: Numeric, D: Data<T>>: Transformer<T, D> {
    /// Get the scale parameters for each feature
    fn scale(&self) -> Option<Vec<T>> {
        None
    }

    /// Get the offset/mean for each feature
    fn offset(&self) -> Option<Vec<T>> {
        None
    }

    /// Inverse transform (undo scaling)
    fn inverse_transform(&self, X: &D) -> Result<Vec<Vec<T>>, String>;
}

/// Trait for encoders (OneHotEncoder, LabelEncoder, etc.)
pub trait Encoder<T: Numeric> {
    /// The input type (e.g., String for categorical data)
    type Input;

    /// Fit the encoder to categorical data
    fn fit(&mut self, data: &[Self::Input]) -> Result<(), String>;

    /// Transform categorical data to numeric
    fn transform(&self, data: &[Self::Input]) -> Result<Vec<Vec<T>>, String>;

    /// Fit and transform in one step
    fn fit_transform(&mut self, data: &[Self::Input]) -> Result<Vec<Vec<T>>, String> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Inverse transform (decode)
    fn inverse_transform(&self, encoded: &[Vec<T>]) -> Result<Vec<Self::Input>, String>;

    /// Get the unique categories
    fn categories(&self) -> Option<Vec<Self::Input>> {
        None
    }
}

/// Trait for imputers (handle missing values)
pub trait Imputer<T: Numeric, D: Data<T>>: Transformer<T, D> {
    /// Get the imputation strategy
    fn strategy(&self) -> ImputationStrategy {
        ImputationStrategy::Mean
    }

    /// Get the statistics used for imputation (e.g., mean, median)
    fn statistics(&self) -> Option<Vec<T>> {
        None
    }
}

/// Imputation strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImputationStrategy {
    /// Replace with mean
    Mean,
    /// Replace with median
    Median,
    /// Replace with most frequent value
    MostFrequent,
    /// Replace with constant
    Constant,
}
