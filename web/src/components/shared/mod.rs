/// Shared UI components for the ML Playground
///
/// This module contains reusable components that provide common functionality
/// across the application, including error handling, validation, and UI primitives.
pub mod algorithm_configurator;
pub mod data_quality;
pub mod data_table;
pub mod error_boundary;
pub mod feature_selector;
pub mod model_performance;
pub mod results_display;
pub mod summary_stats;
pub mod validation;

pub use algorithm_configurator::{
    get_algorithm_parameters, AlgorithmCategory, AlgorithmConfigurator, AlgorithmParameter,
    AlgorithmType, ParameterConstraints, ParameterControl, ParameterType, ParameterValue,
    ValidationResult,
};
pub use data_quality::{analyze_quality, DataQuality, IssueType, QualityIssue, Severity};
pub use data_table::{ColumnConfig, DataTable, SortDirection, TextAlign};
pub use error_boundary::{catch_panic, catch_panic_async, validate_algorithm_input, ErrorBoundary};
pub use feature_selector::{DataType, Feature, FeatureCard, FeatureSelector};
pub use model_performance::{LossChart, ModelPerformanceCard, PerformanceMetrics, TrainingStatus};
pub use results_display::{AlgorithmResults, MetricValue, ResultsDisplay, Visualization};
pub use summary_stats::{
    compute_histogram, ColumnStats, DistributionHistogram, HistogramBin, SummaryStats,
};
pub use validation::{validators, ValidatedInput, ValidationState};
