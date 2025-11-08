//! Clustering algorithms for unsupervised learning
//!
//! This crate provides efficient implementations of popular clustering algorithms
//! including K-means, DBSCAN, and hierarchical clustering.

pub mod kmeans;

pub use kmeans::KMeans;
