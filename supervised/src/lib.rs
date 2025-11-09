//! Supervised learning algorithms
//!
//! This crate provides implementations of supervised learning algorithms
//! including classification and regression models.

pub mod logistic_regression;
pub mod naive_bayes;

pub use logistic_regression::LogisticRegression;
pub use naive_bayes::GaussianNB;
