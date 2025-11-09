//! # ML Traits
//!
//! Core trait definitions for the RustML ecosystem, inspired by NumPy/SciPy's design.
//!
//! This crate defines the foundational traits that enable:
//! - Generic algorithms across different data structures
//! - Composable ML pipelines
//! - Zero-cost abstractions
//! - WASM-compatible implementations

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub mod supervised;
pub mod unsupervised;
pub mod preprocessing;
pub mod metrics;
pub mod clustering;
pub mod reduction;

// Re-export commonly used items
pub use error::MLError;

use std::fmt::Debug;

/// Core trait for numerical types used in ML operations
///
/// This trait enables generic implementations across f32, f64, and potentially
/// custom numeric types while maintaining zero-cost abstractions.
pub trait Numeric:
    Copy
    + Debug
    + Default
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
    /// Zero value for this numeric type
    fn zero() -> Self;

    /// One value for this numeric type
    fn one() -> Self;

    /// Convert from f64
    fn from_f64(value: f64) -> Self;

    /// Convert to f64
    fn to_f64(self) -> f64;

    /// Absolute value
    fn abs(self) -> Self;

    /// Square root
    fn sqrt(self) -> Self;

    /// Power function
    fn powf(self, exp: Self) -> Self;
}

impl Numeric for f32 {
    #[inline]
    fn zero() -> Self { 0.0 }

    #[inline]
    fn one() -> Self { 1.0 }

    #[inline]
    fn from_f64(value: f64) -> Self { value as f32 }

    #[inline]
    fn to_f64(self) -> f64 { self as f64 }

    #[inline]
    fn abs(self) -> Self { f32::abs(self) }

    #[inline]
    fn sqrt(self) -> Self { f32::sqrt(self) }

    #[inline]
    fn powf(self, exp: Self) -> Self { f32::powf(self, exp) }
}

impl Numeric for f64 {
    #[inline]
    fn zero() -> Self { 0.0 }

    #[inline]
    fn one() -> Self { 1.0 }

    #[inline]
    fn from_f64(value: f64) -> Self { value }

    #[inline]
    fn to_f64(self) -> f64 { self }

    #[inline]
    fn abs(self) -> Self { f64::abs(self) }

    #[inline]
    fn sqrt(self) -> Self { f64::sqrt(self) }

    #[inline]
    fn powf(self, exp: Self) -> Self { f64::powf(self, exp) }
}

/// Trait for data that can be used in ML algorithms
///
/// This enables working with different data structures (Matrix, DataFrame, etc.)
/// in a uniform way.
pub trait Data<T: Numeric> {
    /// Get the shape of the data (rows, columns)
    fn shape(&self) -> (usize, usize);

    /// Get a specific element
    fn get(&self, row: usize, col: usize) -> Option<T>;

    /// Get the number of samples (rows)
    fn n_samples(&self) -> usize {
        self.shape().0
    }

    /// Get the number of features (columns)
    fn n_features(&self) -> usize {
        self.shape().1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numeric_f64() {
        let x: f64 = 4.0;
        assert_eq!(x.sqrt(), 2.0);
        assert_eq!(f64::zero(), 0.0);
        assert_eq!(f64::one(), 1.0);
    }

    #[test]
    fn test_numeric_f32() {
        let x: f32 = 4.0;
        assert_eq!(x.sqrt(), 2.0);
        assert_eq!(f32::zero(), 0.0);
        assert_eq!(f32::one(), 1.0);
    }
}
