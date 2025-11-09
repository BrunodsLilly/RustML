//! Data preprocessing transformers
//!
//! This crate provides implementations of common data preprocessing techniques
//! including scaling, normalization, encoding, and imputation.

pub mod scalers;

pub use scalers::{MinMaxScaler, RobustScaler, StandardScaler};
