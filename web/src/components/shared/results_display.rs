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
    let n_classes = matrix.len();
    if n_classes == 0 {
        return rsx! {
            div { class: "confusion-matrix-empty",
                p { "No confusion matrix data available" }
            }
        };
    }

    // Find max value for color scaling
    let max_value = matrix
        .iter()
        .flat_map(|row| row.iter())
        .max()
        .copied()
        .unwrap_or(1);

    // Calculate cell size and dimensions
    let cell_size = 80;
    let label_width = 60;
    let label_height = 40;
    let margin = 20;
    let width = n_classes * cell_size + label_width + margin * 2;
    let height = n_classes * cell_size + label_height + margin * 2;

    // Pre-compute all labels to avoid lifetime issues
    let display_labels: Vec<String> = (0..n_classes)
        .map(|i| {
            class_labels
                .get(i)
                .map(|s| s.clone())
                .unwrap_or_else(|| format!("C{}", i))
        })
        .collect();

    rsx! {
        div { class: "confusion-matrix-viz",
            h4 { "Confusion Matrix" }
            p { class: "viz-subtitle", "Actual vs. Predicted Class Distribution" }

            svg {
                width: "{width}",
                height: "{height}",
                view_box: "0 0 {width} {height}",

                // Title annotations
                text {
                    x: "{label_width + n_classes * cell_size / 2}",
                    y: "{margin - 5}",
                    text_anchor: "middle",
                    class: "confusion-matrix-title",
                    "Predicted Class"
                }

                text {
                    x: "{margin - 10}",
                    y: "{label_height + n_classes * cell_size / 2}",
                    text_anchor: "middle",
                    class: "confusion-matrix-title",
                    transform: "rotate(-90, {margin - 10}, {label_height + n_classes * cell_size / 2})",
                    "Actual Class"
                }

                // Column labels (predicted)
                for (j , label) in display_labels.iter().enumerate() {
                    text {
                        x: "{label_width + j * cell_size + cell_size / 2}",
                        y: "{label_height - 10}",
                        text_anchor: "middle",
                        class: "confusion-matrix-label",
                        "{label}"
                    }
                }

                // Row labels (actual) and cells
                for (i , row_label) in display_labels.iter().enumerate() {
                    // Row label
                    text {
                        x: "{label_width - 10}",
                        y: "{label_height + i * cell_size + cell_size / 2 + 5}",
                        text_anchor: "end",
                        class: "confusion-matrix-label",
                        "{row_label}"
                    }

                    // Cells for this row
                    for j in 0..n_classes {
                        {
                            let value = matrix[i][j];
                            let intensity = if max_value > 0 {
                                (value as f64 / max_value as f64 * 100.0).min(100.0)
                            } else {
                                0.0
                            };

                            // Diagonal cells (correct predictions) use green, off-diagonal use red
                            let (base_color, text_color) = if i == j {
                                ("34, 197, 94", if intensity > 50.0 { "#ffffff" } else { "#1f2937" })
                            } else {
                                ("239, 68, 68", if intensity > 50.0 { "#ffffff" } else { "#1f2937" })
                            };

                            rsx! {
                                // Cell background
                                rect {
                                    x: "{label_width + j * cell_size}",
                                    y: "{label_height + i * cell_size}",
                                    width: "{cell_size}",
                                    height: "{cell_size}",
                                    fill: "rgba({base_color}, {intensity / 100.0})",
                                    stroke: "#e5e7eb",
                                    stroke_width: "1"
                                }

                                // Cell value
                                text {
                                    x: "{label_width + j * cell_size + cell_size / 2}",
                                    y: "{label_height + i * cell_size + cell_size / 2 + 5}",
                                    text_anchor: "middle",
                                    fill: "{text_color}",
                                    class: "confusion-matrix-value",
                                    "{value}"
                                }
                            }
                        }
                    }
                }
            }

            // Legend
            div { class: "confusion-matrix-legend",
                div { class: "legend-item",
                    div { class: "legend-color correct" }
                    span { "Correct Predictions (Diagonal)" }
                }
                div { class: "legend-item",
                    div { class: "legend-color incorrect" }
                    span { "Incorrect Predictions (Off-Diagonal)" }
                }
            }
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
    let n_components = explained_variance.len();
    if n_components == 0 {
        return rsx! {
            div { class: "variance-chart-empty",
                p { "No variance data available" }
            }
        };
    }

    // Chart dimensions
    let width = 600;
    let height = 400;
    let padding_left = 60;
    let padding_right = 40;
    let padding_top = 40;
    let padding_bottom = 60;
    let chart_width = width - padding_left - padding_right;
    let chart_height = height - padding_top - padding_bottom;

    // Find max variance for scaling
    let max_variance = explained_variance
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(1.0);

    // Bar width
    let bar_width = if n_components > 0 {
        (chart_width as f64 / n_components as f64 * 0.8)
            .max(20.0)
            .min(60.0)
    } else {
        40.0
    };
    let bar_spacing = chart_width as f64 / n_components as f64;

    rsx! {
        div { class: "variance-chart-viz",
            h4 { "Variance Explained by Components" }
            p { class: "viz-subtitle", "Individual and Cumulative Variance" }

            svg {
                width: "{width}",
                height: "{height}",
                view_box: "0 0 {width} {height}",

                // Chart background
                rect {
                    x: "{padding_left}",
                    y: "{padding_top}",
                    width: "{chart_width}",
                    height: "{chart_height}",
                    fill: "#f9fafb",
                    stroke: "#e5e7eb"
                }

                // Y-axis labels and grid lines
                for i in 0..=5 {
                    {
                        let value = max_variance * (i as f64 / 5.0);
                        let y = padding_top + chart_height - (chart_height as f64 * (i as f64 / 5.0)) as usize;

                        rsx! {
                            // Grid line
                            line {
                                x1: "{padding_left}",
                                y1: "{y}",
                                x2: "{padding_left + chart_width}",
                                y2: "{y}",
                                stroke: "#e5e7eb",
                                stroke_width: "1"
                            }

                            // Y-axis label
                            text {
                                x: "{padding_left - 10}",
                                y: "{y + 5}",
                                text_anchor: "end",
                                class: "variance-chart-axis-label",
                                "{value:.2}"
                            }
                        }
                    }
                }

                // X-axis
                line {
                    x1: "{padding_left}",
                    y1: "{padding_top + chart_height}",
                    x2: "{padding_left + chart_width}",
                    y2: "{padding_top + chart_height}",
                    stroke: "#374151",
                    stroke_width: "2"
                }

                // Y-axis
                line {
                    x1: "{padding_left}",
                    y1: "{padding_top}",
                    x2: "{padding_left}",
                    y2: "{padding_top + chart_height}",
                    stroke: "#374151",
                    stroke_width: "2"
                }

                // Bars for individual variance
                for (i, &variance) in explained_variance.iter().enumerate() {
                    {
                        let bar_height = if max_variance > 0.0 {
                            (variance / max_variance * chart_height as f64) as usize
                        } else {
                            0
                        };
                        let x = padding_left + (i as f64 * bar_spacing + (bar_spacing - bar_width) / 2.0) as usize;
                        let y = padding_top + chart_height - bar_height;

                        rsx! {
                            // Individual variance bar
                            rect {
                                x: "{x}",
                                y: "{y}",
                                width: "{bar_width as usize}",
                                height: "{bar_height}",
                                fill: "#8b5cf6",
                                opacity: "0.8"
                            }

                            // Component label
                            text {
                                x: "{x + bar_width as usize / 2}",
                                y: "{padding_top + chart_height + 20}",
                                text_anchor: "middle",
                                class: "variance-chart-x-label",
                                "PC{i + 1}"
                            }

                            // Variance percentage
                            text {
                                x: "{x + bar_width as usize / 2}",
                                y: "{y - 5}",
                                text_anchor: "middle",
                                class: "variance-chart-value",
                                "{variance:.1}%"
                            }
                        }
                    }
                }

                // Cumulative variance line
                {
                    let points: Vec<String> = cumulative_variance.iter().enumerate()
                        .map(|(i, &cum_var)| {
                            let x = padding_left + (i as f64 * bar_spacing + bar_spacing / 2.0) as usize;
                            let y_ratio = if max_variance > 0.0 { cum_var / max_variance } else { 0.0 };
                            let y = padding_top + chart_height - (y_ratio * chart_height as f64) as usize;
                            format!("{},{}", x, y)
                        })
                        .collect();

                    let polyline_points = points.join(" ");

                    rsx! {
                        polyline {
                            points: "{polyline_points}",
                            fill: "none",
                            stroke: "#ef4444",
                            stroke_width: "3",
                            opacity: "0.9"
                        }

                        // Points on the line
                        for (i, &cum_var) in cumulative_variance.iter().enumerate() {
                            {
                                let x = padding_left + (i as f64 * bar_spacing + bar_spacing / 2.0) as usize;
                                let y_ratio = if max_variance > 0.0 { cum_var / max_variance } else { 0.0 };
                                let y = padding_top + chart_height - (y_ratio * chart_height as f64) as usize;

                                rsx! {
                                    circle {
                                        cx: "{x}",
                                        cy: "{y}",
                                        r: "4",
                                        fill: "#ef4444"
                                    }
                                }
                            }
                        }
                    }
                }

                // Y-axis title
                text {
                    x: "{padding_left - 45}",
                    y: "{padding_top + chart_height / 2}",
                    text_anchor: "middle",
                    class: "variance-chart-title",
                    transform: "rotate(-90, {padding_left - 45}, {padding_top + chart_height / 2})",
                    "Variance (%)"
                }

                // X-axis title
                text {
                    x: "{padding_left + chart_width / 2}",
                    y: "{padding_top + chart_height + 50}",
                    text_anchor: "middle",
                    class: "variance-chart-title",
                    "Principal Component"
                }
            }

            // Legend
            div { class: "variance-chart-legend",
                div { class: "legend-item",
                    div { class: "legend-color variance-individual" }
                    span { "Individual Variance" }
                }
                div { class: "legend-item",
                    div { class: "legend-color variance-cumulative" }
                    span { "Cumulative Variance" }
                }
            }
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
