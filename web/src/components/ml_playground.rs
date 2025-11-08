//! ML Playground - Interactive ML algorithms with CSV upload
//!
//! This component provides a comprehensive interface for testing all ML algorithms
//! with user-uploaded CSV data.

use dioxus::prelude::*;
use loader::csv_loader::CsvDataset;

/// ML Playground component
#[component]
pub fn MLPlayground() -> Element {
    // State for uploaded dataset
    let mut csv_dataset = use_signal(|| None::<CsvDataset>);
    let mut selected_algorithm = use_signal(|| Algorithm::KMeans);
    let mut result_message = use_signal(|| String::new());
    let mut is_processing = use_signal(|| false);

    rsx! {
        div { class: "ml-playground",
            header { class: "playground-header",
                h1 { "🚀 ML Playground" }
                p { class: "subtitle",
                    "Upload your CSV and explore machine learning algorithms"
                }
            }

            div { class: "playground-container",
                // Left panel: Data upload and algorithm selection
                aside { class: "control-panel",
                    section { class: "upload-section",
                        h2 { "📊 Upload Data" }
                        input {
                            r#type: "file",
                            accept: ".csv",
                            id: "csv-upload",
                            onchange: move |evt| {
                                async move {
                                    if let Some(file_engine) = evt.files() {
                                        let files = file_engine.files();
                                        if let Some(file_name) = files.first() {
                                            if let Some(file_contents) = file_engine.read_file(file_name).await {
                                                if let Ok(content_str) = String::from_utf8(file_contents) {
                                                    // For unsupervised learning, use first column as dummy target
                                                    // We'll only use the features anyway
                                                    let headers: Vec<&str> = content_str.lines().next()
                                                        .unwrap_or("")
                                                        .split(',')
                                                        .collect();

                                                    if headers.is_empty() {
                                                        result_message.set("❌ CSV has no headers".to_string());
                                                    } else {
                                                        let target_col = headers[0];
                                                        match CsvDataset::from_csv(&content_str, target_col) {
                                                            Ok(dataset) => {
                                                                result_message.set(format!(
                                                                    "✅ Loaded {} rows, {} features",
                                                                    dataset.num_samples,
                                                                    dataset.features.cols
                                                                ));
                                                                csv_dataset.set(Some(dataset));
                                                            }
                                                            Err(e) => {
                                                                result_message.set(format!("❌ Error loading CSV: {}", e));
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    result_message.set("❌ Invalid UTF-8 in file".to_string());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        label {
                            r#for: "csv-upload",
                            class: "upload-button",
                            "Choose CSV File"
                        }

                        if let Some(ref dataset) = *csv_dataset.read() {
                            div { class: "dataset-info",
                                h3 { "Dataset Info" }
                                p { "Samples: {dataset.num_samples}" }
                                p { "Features: {dataset.features.cols}" }
                                if !dataset.feature_names.is_empty() {
                                    p { "Column Names: {dataset.feature_names.join(\", \")}" }
                                }
                            }
                        }
                    }

                    section { class: "algorithm-section",
                        h2 { "🧠 Select Algorithm" }

                        AlgorithmButton {
                            algorithm: Algorithm::KMeans,
                            selected: *selected_algorithm.read() == Algorithm::KMeans,
                            onclick: move |_| selected_algorithm.set(Algorithm::KMeans),
                            icon: "🎯",
                            name: "K-Means Clustering",
                            description: "Group similar data points"
                        }

                        AlgorithmButton {
                            algorithm: Algorithm::PCA,
                            selected: *selected_algorithm.read() == Algorithm::PCA,
                            onclick: move |_| selected_algorithm.set(Algorithm::PCA),
                            icon: "📉",
                            name: "PCA",
                            description: "Reduce dimensionality"
                        }

                        AlgorithmButton {
                            algorithm: Algorithm::LogisticRegression,
                            selected: *selected_algorithm.read() == Algorithm::LogisticRegression,
                            onclick: move |_| selected_algorithm.set(Algorithm::LogisticRegression),
                            icon: "🎲",
                            name: "Logistic Regression",
                            description: "Classification model"
                        }

                        AlgorithmButton {
                            algorithm: Algorithm::StandardScaler,
                            selected: *selected_algorithm.read() == Algorithm::StandardScaler,
                            onclick: move |_| selected_algorithm.set(Algorithm::StandardScaler),
                            icon: "⚖️",
                            name: "Standard Scaler",
                            description: "Normalize features"
                        }

                        AlgorithmButton {
                            algorithm: Algorithm::MinMaxScaler,
                            selected: *selected_algorithm.read() == Algorithm::MinMaxScaler,
                            onclick: move |_| selected_algorithm.set(Algorithm::MinMaxScaler),
                            icon: "📏",
                            name: "MinMax Scaler",
                            description: "Scale to [0, 1]"
                        }
                    }

                    if csv_dataset.read().is_some() {
                        button {
                            class: "run-button",
                            disabled: *is_processing.read(),
                            onclick: move |_| {
                                is_processing.set(true);
                                result_message.set(format!("🔄 Running {}...", selected_algorithm.read().name()));
                                // In a real implementation, run the algorithm here
                                is_processing.set(false);
                            },
                            if *is_processing.read() {
                                "Processing..."
                            } else {
                                "▶ Run Algorithm"
                            }
                        }
                    }
                }

                // Right panel: Results and visualization
                main { class: "results-panel",
                    if !result_message.read().is_empty() {
                        div { class: "result-message",
                            "{result_message}"
                        }
                    }

                    if csv_dataset.read().is_some() && !*is_processing.read() {
                        AlgorithmExplanation { algorithm: *selected_algorithm.read() }
                    }

                    if *is_processing.read() {
                        div { class: "loading-spinner",
                            div { class: "spinner" }
                            p { "Processing your data..." }
                        }
                    }
                }
            }

            footer { class: "playground-footer",
                p {
                    "Built with 🦀 Rust + WASM | Powered by "
                    strong { "RustML" }
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Algorithm {
    KMeans,
    PCA,
    LogisticRegression,
    StandardScaler,
    MinMaxScaler,
}

impl Algorithm {
    fn name(&self) -> &str {
        match self {
            Algorithm::KMeans => "K-Means Clustering",
            Algorithm::PCA => "PCA",
            Algorithm::LogisticRegression => "Logistic Regression",
            Algorithm::StandardScaler => "Standard Scaler",
            Algorithm::MinMaxScaler => "MinMax Scaler",
        }
    }
}

#[component]
fn AlgorithmButton(
    algorithm: Algorithm,
    selected: bool,
    onclick: EventHandler<MouseEvent>,
    icon: &'static str,
    name: &'static str,
    description: &'static str,
) -> Element {
    rsx! {
        button {
            class: if selected { "algorithm-btn selected" } else { "algorithm-btn" },
            onclick: move |evt| onclick.call(evt),
            div { class: "algorithm-icon", "{icon}" }
            div { class: "algorithm-info",
                h3 { "{name}" }
                p { "{description}" }
            }
        }
    }
}

#[component]
fn AlgorithmExplanation(algorithm: Algorithm) -> Element {
    rsx! {
        div { class: "algorithm-explanation",
            match algorithm {
                Algorithm::KMeans => rsx! {
                    div {
                        h2 { "🎯 K-Means Clustering" }
                        p { "Groups your data into K clusters based on similarity." }
                        h3 { "How it works:" }
                        ol {
                            li { "Choose number of clusters (K)" }
                            li { "Algorithm finds cluster centers" }
                            li { "Assigns each point to nearest center" }
                            li { "Iteratively improves cluster quality" }
                        }
                        div { class: "tech-note",
                            strong { "Implementation: " }
                            "K-means++ initialization for better convergence"
                        }
                    }
                },
                Algorithm::PCA => rsx! {
                    div {
                        h2 { "📉 Principal Component Analysis" }
                        p { "Reduces dimensionality while preserving variance." }
                        h3 { "How it works:" }
                        ol {
                            li { "Standardizes your features" }
                            li { "Computes correlation matrix" }
                            li { "Finds principal components (eigenvectors)" }
                            li { "Projects data onto top components" }
                        }
                        div { class: "tech-note",
                            strong { "Implementation: " }
                            "SVD-based with power iteration"
                        }
                    }
                },
                Algorithm::LogisticRegression => rsx! {
                    div {
                        h2 { "🎲 Logistic Regression" }
                        p { "Predicts class probabilities using sigmoid function." }
                        h3 { "How it works:" }
                        ol {
                            li { "Learns decision boundary from labeled data" }
                            li { "Uses gradient descent optimization" }
                            li { "Outputs probability for each class" }
                            li { "Supports binary and multi-class problems" }
                        }
                        div { class: "tech-note",
                            strong { "Implementation: " }
                            "One-vs-rest for multi-class"
                        }
                    }
                },
                Algorithm::StandardScaler => rsx! {
                    div {
                        h2 { "⚖️ Standard Scaler (Z-score)" }
                        p { "Standardizes features to zero mean and unit variance." }
                        h3 { "Formula:" }
                        code { "z = (x - μ) / σ" }
                        h3 { "When to use:" }
                        ul {
                            li { "Before distance-based algorithms" }
                            li { "When features have different scales" }
                            li { "With gradient descent optimization" }
                        }
                        div { class: "tech-note",
                            strong { "Implementation: " }
                            "Single-pass statistics computation"
                        }
                    }
                },
                Algorithm::MinMaxScaler => rsx! {
                    div {
                        h2 { "📏 MinMax Scaler" }
                        p { "Scales features to a fixed range [0, 1]." }
                        h3 { "Formula:" }
                        code { "x_scaled = (x - min) / (max - min)" }
                        h3 { "When to use:" }
                        ul {
                            li { "When you need bounded values" }
                            li { "For neural network inputs" }
                            li { "When zero has special meaning" }
                        }
                        div { class: "tech-note",
                            strong { "Implementation: " }
                            "Preserves zero for sparse data"
                        }
                    }
                },
            }
        }
    }
}
