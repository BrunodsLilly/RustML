/// Shared UI components for the ML Playground
///
/// This module contains reusable components that provide common functionality
/// across the application, including error handling, validation, and UI primitives.

pub mod data_quality;
pub mod data_table;
pub mod error_boundary;
pub mod feature_selector;
pub mod summary_stats;
pub mod validation;

pub use data_quality::{DataQuality, QualityIssue, IssueType, Severity, analyze_quality};
pub use data_table::{DataTable, ColumnConfig, SortDirection, TextAlign};
pub use error_boundary::{ErrorBoundary, catch_panic, catch_panic_async, validate_algorithm_input};
pub use feature_selector::{FeatureSelector, Feature, DataType, FeatureCard};
pub use summary_stats::{SummaryStats, ColumnStats, HistogramBin, DistributionHistogram, compute_histogram};
pub use validation::{ValidationState, ValidatedInput, validators};
