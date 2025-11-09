//! Interactive Results Display Component for ML Playground
//!
//! Provides a comprehensive tabbed interface for displaying machine learning
//! algorithm results with visualizations, metrics, and predictions.

use dioxus::prelude::*;
use std::collections::HashMap;

/// Structured results from running an ML algorithm
#[derive(Clone, Debug, PartialEq)]
pub struct AlgorithmResults {
    /// Algorithm name (e.g., "Naive Bayes", "K-Means")
    pub algorithm_name: String,

    /// Overall accuracy or performance metric (0-100%)
    pub accuracy: Option<f64>,

    /// Predicted values for each sample
    pub predictions: Vec<usize>,

    /// Actual target values (for supervised learning)
    pub actual_values: Option<Vec<f64>>,

    /// Additional metrics specific to the algorithm
    pub metrics: HashMap<String, MetricValue>,

    /// Algorithm-specific visualizations
    pub visualizations: Vec<Visualization>,

    /// Raw model details for expert users
    pub model_details: HashMap<String, String>,
}

/// A metric value that can be numeric or string
#[derive(Clone, Debug, PartialEq)]
pub enum MetricValue {
    Numeric(f64),
    Integer(usize),
    Text(String),
    Percentage(f64),
}

impl MetricValue {
    pub fn display(&self) -> String {
        match self {
            MetricValue::Numeric(v) => format!("{:.4}", v),
            MetricValue::Integer(v) => v.to_string(),
            MetricValue::Text(s) => s.clone(),
            MetricValue::Percentage(v) => format!("{:.1}%", v),
        }
    }
}

/// Visualization types supported by the results display
#[derive(Clone, Debug, PartialEq)]
pub enum Visualization {
    /// Confusion matrix for classification
    ConfusionMatrix {
        matrix: Vec<Vec<usize>>,
        class_labels: Vec<String>,
    },
    /// Cluster distribution for K-Means
    ClusterDistribution {
        cluster_sizes: Vec<usize>,
        cluster_centers: Option<Vec<Vec<f64>>>,
    },
    /// Variance explained for PCA
    VarianceExplained {
        explained_variance: Vec<f64>,
        cumulative_variance: Vec<f64>,
    },
    /// Scaler statistics (before/after)
    ScalerStats {
        before_mean: Vec<f64>,
        before_std: Vec<f64>,
        after_mean: Vec<f64>,
        after_std: Vec<f64>,
        feature_names: Vec<String>,
    },
}

/// Main results display component with tabbed interface
#[component]
pub fn ResultsDisplay(results: ReadOnlySignal<AlgorithmResults>) -> Element {
    let mut active_tab = use_signal(|| "overview");

    let results_val = results();

    rsx! {
        div { class: "results-display",
            // Header
            div { class: "results-header",
                h2 { "📊 {results_val.algorithm_name} Results" }
                if let Some(accuracy) = results_val.accuracy {
                    p { class: "accuracy-badge",
                        "Accuracy: {accuracy:.1}%"
                    }
                }
            }

            // Tab navigation
            div { class: "tab-navigation",
                button {
                    class: if active_tab() == "overview" { "tab-button active" } else { "tab-button" },
                    onclick: move |_| active_tab.set("overview"),
                    "📋 Overview"
                }
                button {
                    class: if active_tab() == "predictions" { "tab-button active" } else { "tab-button" },
                    onclick: move |_| active_tab.set("predictions"),
                    "🎯 Predictions"
                }
                button {
                    class: if active_tab() == "metrics" { "tab-button active" } else { "tab-button" },
                    onclick: move |_| active_tab.set("metrics"),
                    "📈 Metrics"
                }
                if !results_val.visualizations.is_empty() {
                    button {
                        class: if active_tab() == "visualizations" { "tab-button active" } else { "tab-button" },
                        onclick: move |_| active_tab.set("visualizations"),
                        "📊 Visualizations"
                    }
                }
            }

            // Tab content
            div { class: "tab-content",
                // Overview Tab
                if active_tab() == "overview" {
                    OverviewTab { results: results_val.clone() }
                }

                // Predictions Tab
                if active_tab() == "predictions" {
                    PredictionsTab { results: results_val.clone() }
                }

                // Metrics Tab
                if active_tab() == "metrics" {
                    MetricsTab { results: results_val.clone() }
                }

                // Visualizations Tab
                if active_tab() == "visualizations" && !results_val.visualizations.is_empty() {
                    VisualizationsTab { results: results_val.clone() }
                }
            }
        }
    }
}

/// Overview tab - summary metrics cards
#[component]
fn OverviewTab(results: AlgorithmResults) -> Element {
    rsx! {
        div { class: "overview-tab",
            h3 { "Summary Metrics" }

            div { class: "metrics-grid",
                // Accuracy card (if available)
                if let Some(accuracy) = results.accuracy {
                    div { class: "metric-card accuracy",
                        div { class: "metric-icon", "🎯" }
                        div { class: "metric-content",
                            p { class: "metric-label", "Accuracy" }
                            p { class: "metric-value", "{accuracy:.1}%" }
                        }
                    }
                }

                // Predictions count
                div { class: "metric-card",
                    div { class: "metric-icon", "📊" }
                    div { class: "metric-content",
                        p { class: "metric-label", "Predictions" }
                        p { class: "metric-value", "{results.predictions.len()}" }
                    }
                }

                // Additional metrics
                for (key , value) in results.metrics.iter() {
                    div { class: "metric-card",
                        div { class: "metric-icon", "📈" }
                        div { class: "metric-content",
                            p { class: "metric-label", "{key}" }
                            p { class: "metric-value", "{value.display()}" }
                        }
                    }
                }
            }

            // Model Details
            if !results.model_details.is_empty() {
                div { class: "model-details",
                    h4 { "Model Details" }
                    for (key , value) in results.model_details.iter() {
                        div { class: "detail-row",
                            span { class: "detail-label", "{key}:" }
                            span { class: "detail-value", "{value}" }
                        }
                    }
                }
            }
        }
    }
}

/// Predictions tab - paginated table of predictions vs actual
#[component]
fn PredictionsTab(results: AlgorithmResults) -> Element {
    let mut current_page = use_signal(|| 0);
    const ROWS_PER_PAGE: usize = 50;

    let total_pages = (results.predictions.len() + ROWS_PER_PAGE - 1) / ROWS_PER_PAGE;
    let start_idx = current_page() * ROWS_PER_PAGE;
    let end_idx = (start_idx + ROWS_PER_PAGE).min(results.predictions.len());

    rsx! {
        div { class: "predictions-tab",
            h3 { "Predictions Table" }
            p { class: "subtitle", "Showing {start_idx + 1}-{end_idx} of {results.predictions.len()} predictions" }

            div { class: "predictions-table-container",
                table { class: "predictions-table",
                    thead {
                        tr {
                            th { "Row" }
                            th { "Predicted" }
                            if results.actual_values.is_some() {
                                th { "Actual" }
                                th { "Correct?" }
                            }
                        }
                    }
                    tbody {
                        for i in start_idx..end_idx {
                            {
                                let prediction = results.predictions[i];
                                let actual = results.actual_values.as_ref().map(|v| v[i]);
                                let is_correct = actual.map(|a| a as usize == prediction);

                                rsx! {
                                    tr {
                                        class: if is_correct == Some(false) { "incorrect" } else { "" },
                                        td { "{i + 1}" }
                                        td { "{prediction}" }
                                        if let Some(actual_val) = actual {
                                            td { "{actual_val}" }
                                            td {
                                                if is_correct == Some(true) {
                                                    span { class: "correct-badge", "✓" }
                                                } else {
                                                    span { class: "incorrect-badge", "✗" }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Pagination controls
            if total_pages > 1 {
                div { class: "pagination",
                    button {
                        disabled: current_page() == 0,
                        onclick: move |_| current_page.set(current_page().saturating_sub(1)),
                        "← Previous"
                    }
                    span { class: "page-info",
                        "Page {current_page() + 1} of {total_pages}"
                    }
                    button {
                        disabled: current_page() >= total_pages - 1,
                        onclick: move |_| if current_page() < total_pages - 1 { current_page.set(current_page() + 1) },
                        "Next →"
                    }
                }
            }
        }
    }
}

/// Metrics tab - detailed metrics display
#[component]
fn MetricsTab(results: AlgorithmResults) -> Element {
    rsx! {
        div { class: "metrics-tab",
            h3 { "Detailed Metrics" }

            div { class: "metrics-list",
                for (key , value) in results.metrics.iter() {
                    div { class: "metric-row",
                        div { class: "metric-name", "{key}" }
                        div { class: "metric-value-display", "{value.display()}" }
                    }
                }
            }

            if results.metrics.is_empty() {
                p { class: "no-metrics", "No additional metrics available for this algorithm." }
            }
        }
    }
}

/// Visualizations tab - algorithm-specific charts
#[component]
fn VisualizationsTab(results: AlgorithmResults) -> Element {
    rsx! {
        div { class: "visualizations-tab",
            h3 { "Visualizations" }

            for viz in results.visualizations.iter() {
                match viz {
                    Visualization::ConfusionMatrix { matrix, class_labels } => {
                        rsx! {
                            div { class: "visualization-container",
                                h4 { "Confusion Matrix" }
                                ConfusionMatrixViz {
                                    matrix: matrix.clone(),
                                    class_labels: class_labels.clone(),
                                }
                            }
                        }
                    }
                    Visualization::ClusterDistribution { cluster_sizes, .. } => {
                        rsx! {
                            div { class: "visualization-container",
                                h4 { "Cluster Distribution" }
                                ClusterDistributionViz {
                                    cluster_sizes: cluster_sizes.clone(),
                                }
                            }
                        }
                    }
                    Visualization::VarianceExplained { explained_variance, cumulative_variance } => {
                        rsx! {
                            div { class: "visualization-container",
                                h4 { "Variance Explained" }
                                VarianceChartViz {
                                    explained_variance: explained_variance.clone(),
                                    cumulative_variance: cumulative_variance.clone(),
                                }
                            }
                        }
                    }
                    Visualization::ScalerStats { before_mean, after_mean, feature_names, .. } => {
                        rsx! {
                            div { class: "visualization-container",
                                h4 { "Scaler Statistics" }
                                ScalerStatsViz {
                                    before_mean: before_mean.clone(),
                                    after_mean: after_mean.clone(),
                                    feature_names: feature_names.clone(),
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Confusion Matrix visualization component
#[component]
fn ConfusionMatrixViz(matrix: Vec<Vec<usize>>, class_labels: Vec<String>) -> Element {
    rsx! {
        div { class: "confusion-matrix",
            p { "Confusion matrix visualization coming soon..." }
            // TODO: Implement SVG heatmap
        }
    }
}

/// Cluster distribution visualization
#[component]
fn ClusterDistributionViz(cluster_sizes: Vec<usize>) -> Element {
    rsx! {
        div { class: "cluster-distribution",
            for (i , size) in cluster_sizes.iter().enumerate() {
                div { class: "cluster-bar",
                    div { class: "cluster-label", "Cluster {i}" }
                    div {
                        class: "cluster-bar-fill",
                        style: "width: {(*size as f64 / cluster_sizes.iter().sum::<usize>() as f64) * 100.0}%",
                        "{size} samples"
                    }
                }
            }
        }
    }
}

/// Variance explained chart for PCA
#[component]
fn VarianceChartViz(explained_variance: Vec<f64>, cumulative_variance: Vec<f64>) -> Element {
    rsx! {
        div { class: "variance-chart",
            p { "Variance chart visualization coming soon..." }
            // TODO: Implement chart
        }
    }
}

/// Scaler statistics visualization
#[component]
fn ScalerStatsViz(
    before_mean: Vec<f64>,
    after_mean: Vec<f64>,
    feature_names: Vec<String>,
) -> Element {
    rsx! {
        div { class: "scaler-stats",
            table {
                thead {
                    tr {
                        th { "Feature" }
                        th { "Before Mean" }
                        th { "After Mean" }
                    }
                }
                tbody {
                    for (i , name) in feature_names.iter().enumerate() {
                        tr {
                            td { "{name}" }
                            td { "{before_mean[i]:.2}" }
                            td { "{after_mean[i]:.2}" }
                        }
                    }
                }
            }
        }
    }
}
