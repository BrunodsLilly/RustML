//! ML Playground - Interactive ML algorithms with CSV upload
//!
//! This component provides a comprehensive interface for testing all ML algorithms
//! with user-uploaded CSV data.

use clustering::kmeans::KMeans;
use decision_tree::{DecisionTreeClassifier, SplitCriterion};
use dimensionality_reduction::pca::PCA;
use dioxus::prelude::*;
use loader::csv_loader::CsvDataset;
use ml_traits::clustering::Clusterer;
use ml_traits::preprocessing::Transformer;
use ml_traits::supervised::SupervisedModel;
use ml_traits::unsupervised::UnsupervisedModel;
use preprocessing::scalers::{MinMaxScaler, StandardScaler};
use supervised::logistic_regression::LogisticRegression;
use supervised::naive_bayes::GaussianNB;

use crate::components::shared::{
    AlgorithmCategory, AlgorithmConfigurator, AlgorithmParameter, AlgorithmType,
    ModelPerformanceCard, PerformanceMetrics, TrainingStatus,
};

/// ML Playground component
#[component]
pub fn MLPlayground() -> Element {
    // State for uploaded dataset
    let mut csv_dataset = use_signal(|| None::<CsvDataset>);
    let mut selected_algorithm = use_signal(|| Algorithm::KMeans);
    let mut result_message = use_signal(|| String::new());
    let mut is_processing = use_signal(|| false);

    // State for algorithm configuration
    let mut show_configurator = use_signal(|| false);
    let mut algorithm_params = use_signal(|| AlgorithmParams::default());

    // State for training metrics
    let mut performance_metrics = use_signal(|| None::<PerformanceMetrics>);
    let mut show_performance = use_signal(|| false);

    // State for train/test split
    let mut train_split_pct = use_signal(|| 80); // Default 80% train, 20% test
    let mut show_predictions = use_signal(|| false);
    let mut predictions_data = use_signal(|| None::<PredictionsData>);

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
                                                // Input validation: file size limit (5MB max)
                                                const MAX_FILE_SIZE: usize = 5 * 1024 * 1024; // 5MB
                                                if file_contents.len() > MAX_FILE_SIZE {
                                                    result_message.set(format!(
                                                        "❌ File too large: {:.2} MB (max 5MB)\n\n\
                                                        Large datasets can crash the browser.\n\
                                                        Try filtering your data or using a smaller sample.",
                                                        file_contents.len() as f64 / (1024.0 * 1024.0)
                                                    ));
                                                    return;
                                                }

                                                if let Ok(content_str) = String::from_utf8(file_contents) {
                                                    // Input validation: row and column limits
                                                    const MAX_ROWS: usize = 10000;
                                                    const MAX_COLS: usize = 100;

                                                    let line_count = content_str.lines().count();
                                                    if line_count > MAX_ROWS + 1 { // +1 for header
                                                        result_message.set(format!(
                                                            "❌ Too many rows: {} (max {})\n\n\
                                                            Try sampling your dataset first.",
                                                            line_count - 1, MAX_ROWS
                                                        ));
                                                        return;
                                                    }

                                                    // For unsupervised learning, use first column as dummy target
                                                    // We'll only use the features anyway
                                                    let headers: Vec<&str> = content_str.lines().next()
                                                        .unwrap_or("")
                                                        .split(',')
                                                        .collect();

                                                    if headers.is_empty() {
                                                        result_message.set("❌ CSV has no headers".to_string());
                                                    } else if headers.len() > MAX_COLS {
                                                        result_message.set(format!(
                                                            "❌ Too many columns: {} (max {})\n\n\
                                                            Consider reducing feature count with PCA first.",
                                                            headers.len(), MAX_COLS
                                                        ));
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

                            // Train/Test Split Configuration
                            div { class: "train-test-split",
                                h3 { "📊 Data Split" }
                                p { class: "split-description",
                                    "Split your data into training and testing sets for model evaluation"
                                }

                                div { class: "split-control",
                                    label {
                                        r#for: "train-split-slider",
                                        "Training Data: {train_split_pct}%"
                                    }
                                    input {
                                        r#type: "range",
                                        id: "train-split-slider",
                                        min: "50",
                                        max: "90",
                                        step: "5",
                                        value: "{train_split_pct}",
                                        oninput: move |evt| {
                                            if let Ok(val) = evt.value().parse::<i32>() {
                                                train_split_pct.set(val);
                                            }
                                        }
                                    }
                                }

                                div { class: "split-summary",
                                    {
                                        let train_pct = *train_split_pct.read();
                                        let train_count = (dataset.num_samples * (train_pct as usize)) / 100;
                                        let test_count = dataset.num_samples - train_count;
                                        let test_pct = 100 - train_pct;
                                        rsx! {
                                            div { class: "split-info train-info",
                                                span { class: "split-label", "Train:" }
                                                span { class: "split-value",
                                                    "{train_count} samples ({train_pct}%)"
                                                }
                                            }
                                            div { class: "split-info test-info",
                                                span { class: "split-label", "Test:" }
                                                span { class: "split-value",
                                                    "{test_count} samples ({test_pct}%)"
                                                }
                                            }
                                        }
                                    }
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
                            algorithm: Algorithm::DecisionTree,
                            selected: *selected_algorithm.read() == Algorithm::DecisionTree,
                            onclick: move |_| selected_algorithm.set(Algorithm::DecisionTree),
                            icon: "🌳",
                            name: "Decision Tree",
                            description: "Tree-based classifier"
                        }

                        AlgorithmButton {
                            algorithm: Algorithm::NaiveBayes,
                            selected: *selected_algorithm.read() == Algorithm::NaiveBayes,
                            onclick: move |_| selected_algorithm.set(Algorithm::NaiveBayes),
                            icon: "🎯",
                            name: "Naive Bayes",
                            description: "Probabilistic classifier"
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
                        div { class: "action-buttons",
                            button {
                                class: "config-button",
                                onclick: move |_| {
                                    let current = *show_configurator.read();
                                    show_configurator.set(!current);
                                },
                                "⚙️ Configure Parameters"
                            }

                            button {
                                class: "run-button",
                                disabled: *is_processing.read(),
                                onclick: move |_| {
                                    spawn(async move {
                                        is_processing.set(true);
                                        show_performance.set(true);
                                        result_message.set(format!("🔄 Running {}...", selected_algorithm.read().name()));

                                        // Initialize performance metrics
                                        let max_iter = match *selected_algorithm.read() {
                                            Algorithm::KMeans => algorithm_params.read().kmeans_max_iter,
                                            Algorithm::LogisticRegression => algorithm_params.read().logreg_max_iter,
                                            _ => 100,
                                        };

                                        performance_metrics.set(Some(PerformanceMetrics::new(max_iter)));

                                        if let Some(ref dataset) = *csv_dataset.read() {
                                            // WASM panic boundary: catch crashes gracefully
                                            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                                run_algorithm_with_metrics(
                                                    *selected_algorithm.read(),
                                                    dataset,
                                                    &algorithm_params.read(),
                                                    &mut performance_metrics,
                                                    *train_split_pct.read() as usize,
                                                    &mut predictions_data,
                                                    &mut show_predictions
                                                )
                                            }));

                                            match result {
                                                Ok(msg) => result_message.set(msg),
                                                Err(panic_info) => {
                                                    // Log panic details for debugging
                                                    web_sys::console::error_1(&"❌ WASM panic caught during algorithm execution".into());
                                                    web_sys::console::error_1(&format!("{:?}", panic_info).into());

                                                    result_message.set(format!(
                                                        "❌ Algorithm crashed unexpectedly.\n\n\
                                                        This can happen when:\n\
                                                        • Dataset is too large (try <1000 rows)\n\
                                                        • Features have invalid values (NaN, Infinity)\n\
                                                        • Parameters are out of valid range\n\n\
                                                        💡 Try:\n\
                                                        • Reducing dataset size\n\
                                                        • Checking for missing/invalid data\n\
                                                        • Using different parameter values\n\n\
                                                        Check browser console for technical details."
                                                    ));
                                                }
                                            }
                                        }

                                        is_processing.set(false);
                                    });
                                },
                                if *is_processing.read() {
                                    "Processing..."
                                } else {
                                    "▶ Run Algorithm"
                                }
                            }
                        }
                    }

                    // Algorithm Configurator (show when toggled)
                    {
                        let show_config = *show_configurator.read();
                        if show_config {
                            let algo_type = selected_algorithm.read().to_algorithm_type();
                            rsx! {
                                div { class: "configurator-panel",
                                    AlgorithmConfigurator {
                                        algorithm: algo_type,
                                        on_parameters_change: move |params: Vec<AlgorithmParameter>| {
                                            // Update algorithm_params based on received parameters
                                            let mut current_params = algorithm_params.write();

                                            // Update parameters based on algorithm type
                                            for param in params {
                                                match param.name.as_str() {
                                                    "n_clusters" => if let Some(val) = param.current_value.as_i64() {
                                                        current_params.k_clusters = val as usize;
                                                    },
                                                    "max_iterations" => if let Some(val) = param.current_value.as_i64() {
                                                        match algo_type {
                                                            AlgorithmType::KMeans => current_params.kmeans_max_iter = val as usize,
                                                            AlgorithmType::LogisticRegression => current_params.logreg_max_iter = val as usize,
                                                            _ => {}
                                                        }
                                                    },
                                                    "tolerance" => if let Some(val) = param.current_value.as_f64() {
                                                        match algo_type {
                                                            AlgorithmType::KMeans => current_params.kmeans_tolerance = val,
                                                            AlgorithmType::LogisticRegression => current_params.logreg_tolerance = val,
                                                            _ => {}
                                                        }
                                                    },
                                                    "n_components" => if let Some(val) = param.current_value.as_i64() {
                                                        current_params.n_components = val as usize;
                                                    },
                                                    "learning_rate" => if let Some(val) = param.current_value.as_f64() {
                                                        current_params.learning_rate = val;
                                                    },
                                                    "max_depth" => if let Some(val) = param.current_value.as_i64() {
                                                        current_params.dt_max_depth = val as usize;
                                                    },
                                                    "min_samples_split" => if let Some(val) = param.current_value.as_i64() {
                                                        current_params.dt_min_samples_split = val as usize;
                                                    },
                                                    "min_samples_leaf" => if let Some(val) = param.current_value.as_i64() {
                                                        current_params.dt_min_samples_leaf = val as usize;
                                                    },
                                                    "min_value" => if let Some(val) = param.current_value.as_f64() {
                                                        current_params.scaler_min = val;
                                                    },
                                                    "max_value" => if let Some(val) = param.current_value.as_f64() {
                                                        current_params.scaler_max = val;
                                                    },
                                                    _ => {}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            rsx! { div {} }
                        }
                    }
                }

                // Right panel: Results and visualization
                main { class: "results-panel",
                    // Performance metrics card (shown during/after training)
                    {
                        let show_perf = *show_performance.read();
                        if show_perf {
                            if let Some(metrics) = performance_metrics.read().clone() {
                                let algo_name = selected_algorithm.read().name().to_string();
                                rsx! {
                                    ModelPerformanceCard {
                                        metrics,
                                        model_name: algo_name,
                                        show_loss_chart: true,
                                        show_details: true,
                                    }
                                }
                            } else {
                                rsx! { div {} }
                            }
                        } else {
                            rsx! { div {} }
                        }
                    }

                    if !result_message.read().is_empty() {
                        div { class: "result-message",
                            "{result_message}"
                        }
                    }

                    // Predictions Table (shown after supervised learning)
                    if *show_predictions.read() {
                        if let Some(ref pred_data) = *predictions_data.read() {
                            PredictionsTable { predictions: pred_data.clone() }
                        }
                    }

                    if csv_dataset.read().is_some() && !*is_processing.read() && !*show_performance.read() {
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
    DecisionTree,
    NaiveBayes,
    StandardScaler,
    MinMaxScaler,
}

impl Algorithm {
    fn name(&self) -> &str {
        match self {
            Algorithm::KMeans => "K-Means Clustering",
            Algorithm::PCA => "PCA",
            Algorithm::LogisticRegression => "Logistic Regression",
            Algorithm::DecisionTree => "Decision Tree",
            Algorithm::NaiveBayes => "Naive Bayes",
            Algorithm::StandardScaler => "Standard Scaler",
            Algorithm::MinMaxScaler => "MinMax Scaler",
        }
    }

    fn to_algorithm_type(&self) -> AlgorithmType {
        match self {
            Algorithm::KMeans => AlgorithmType::KMeans,
            Algorithm::PCA => AlgorithmType::PCA,
            Algorithm::LogisticRegression => AlgorithmType::LogisticRegression,
            Algorithm::DecisionTree => AlgorithmType::DecisionTree,
            Algorithm::NaiveBayes => AlgorithmType::NaiveBayes,
            Algorithm::StandardScaler => AlgorithmType::StandardScaler,
            Algorithm::MinMaxScaler => AlgorithmType::MinMaxScaler,
        }
    }
}

/// Parameters for algorithm configuration
#[derive(Debug, Clone, PartialEq)]
struct AlgorithmParams {
    // K-Means
    k_clusters: usize,
    kmeans_max_iter: usize,
    kmeans_tolerance: f64,

    // PCA
    n_components: usize,

    // Logistic Regression
    learning_rate: f64,
    logreg_max_iter: usize,
    logreg_tolerance: f64,

    // Decision Tree
    dt_max_depth: usize,
    dt_min_samples_split: usize,
    dt_min_samples_leaf: usize,

    // Scalers
    scaler_min: f64,
    scaler_max: f64,
}

impl Default for AlgorithmParams {
    fn default() -> Self {
        Self {
            k_clusters: 3,
            kmeans_max_iter: 100,
            kmeans_tolerance: 1e-4,
            n_components: 2,
            learning_rate: 0.01,
            logreg_max_iter: 1000,
            logreg_tolerance: 1e-4,
            dt_max_depth: 10,
            dt_min_samples_split: 2,
            dt_min_samples_leaf: 1,
            scaler_min: 0.0,
            scaler_max: 1.0,
        }
    }
}

/// Predictions data for supervised learning
#[derive(Debug, Clone, PartialEq)]
struct PredictionsData {
    // Train set results
    train_actual: Vec<usize>,
    train_predicted: Vec<usize>,
    train_accuracy: f64,

    // Test set results
    test_actual: Vec<usize>,
    test_predicted: Vec<usize>,
    test_accuracy: f64,

    // Overall metrics
    num_classes: usize,
    confusion_matrix: Vec<Vec<usize>>, // confusion_matrix[actual][predicted]
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
                Algorithm::DecisionTree => rsx! {
                    div {
                        h2 { "🌳 Decision Tree" }
                        p { "Creates a tree of decisions to classify data." }
                        h3 { "How it works:" }
                        ol {
                            li { "Recursively splits data based on features" }
                            li { "Chooses splits that maximize information gain" }
                            li { "Creates tree structure with decision rules" }
                            li { "Makes predictions by traversing the tree" }
                        }
                        div { class: "tech-note",
                            strong { "Implementation: " }
                            "CART algorithm with Gini impurity"
                        }
                    }
                },
                Algorithm::NaiveBayes => rsx! {
                    div {
                        h2 { "🎯 Naive Bayes" }
                        p { "Probabilistic classifier using Bayes' theorem." }
                        h3 { "How it works:" }
                        ol {
                            li { "Assumes features are independent (naive assumption)" }
                            li { "Calculates class probabilities using Bayes' theorem" }
                            li { "Models feature distributions as Gaussians" }
                            li { "Predicts class with highest probability" }
                        }
                        div { class: "tech-note",
                            strong { "Implementation: " }
                            "Gaussian Naive Bayes with numerical stability"
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

/// Predictions Table Component - displays train/test predictions with accuracy
#[component]
fn PredictionsTable(predictions: PredictionsData) -> Element {
    rsx! {
        div { class: "predictions-container",
            h2 { "🎯 Predictions & Results" }

            // Overall Summary Stats
            div { class: "summary-stats",
                h3 { "📊 Model Performance" }

                div { class: "stats-grid",
                    div { class: "stat-card train-stat",
                        div { class: "stat-label", "Training Accuracy" }
                        div { class: "stat-value", "{predictions.train_accuracy:.1}%" }
                        div { class: "stat-detail", "{predictions.train_actual.len()} samples" }
                    }

                    div { class: "stat-card test-stat",
                        div { class: "stat-label", "Test Accuracy" }
                        div { class: "stat-value", "{predictions.test_accuracy:.1}%" }
                        div { class: "stat-detail", "{predictions.test_actual.len()} samples" }
                    }

                    div { class: "stat-card classes-stat",
                        div { class: "stat-label", "Classes" }
                        div { class: "stat-value", "{predictions.num_classes}" }
                        div { class: "stat-detail", "unique labels" }
                    }
                }
            }

            // Confusion Matrix
            div { class: "confusion-matrix-section",
                h3 { "🔢 Confusion Matrix (Test Set)" }
                ConfusionMatrix { matrix: predictions.confusion_matrix.clone(), num_classes: predictions.num_classes }
            }

            // Test Predictions Table
            div { class: "predictions-table-section",
                h3 { "📋 Test Set Predictions (First 50)" }

                table { class: "predictions-table",
                    thead {
                        tr {
                            th { "Sample #" }
                            th { "Actual" }
                            th { "Predicted" }
                            th { "Result" }
                        }
                    }
                    tbody {
                        for (i, (&actual, &predicted)) in predictions.test_actual.iter()
                            .zip(predictions.test_predicted.iter())
                            .enumerate()
                            .take(50) {
                            tr {
                                class: if actual == predicted { "correct-prediction" } else { "incorrect-prediction" },
                                td { "{i + 1}" }
                                td { class: "actual-value", "{actual}" }
                                td { class: "predicted-value", "{predicted}" }
                                td {
                                    if actual == predicted {
                                        span { class: "result-icon correct", "✓" }
                                    } else {
                                        span { class: "result-icon incorrect", "✗" }
                                    }
                                }
                            }
                        }
                    }
                }

                if predictions.test_actual.len() > 50 {
                    p { class: "table-note",
                        "Showing first 50 of {predictions.test_actual.len()} test samples"
                    }
                }
            }
        }
    }
}

/// Confusion Matrix Component
#[component]
fn ConfusionMatrix(matrix: Vec<Vec<usize>>, num_classes: usize) -> Element {
    rsx! {
        table { class: "confusion-matrix",
            thead {
                tr {
                    th { class: "matrix-corner", "" }
                    th { colspan: "{num_classes}", class: "predicted-header", "Predicted Class" }
                }
                tr {
                    th { "Actual" }
                    for j in 0..num_classes {
                        th { "Class {j}" }
                    }
                }
            }
            tbody {
                for i in 0..num_classes {
                    tr {
                        th { "Class {i}" }
                        for j in 0..num_classes {
                            td {
                                class: if i == j { "matrix-cell diagonal" } else { "matrix-cell off-diagonal" },
                                "{matrix[i][j]}"
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Run algorithm with performance metrics tracking
fn run_algorithm_with_metrics(
    algorithm: Algorithm,
    dataset: &CsvDataset,
    params: &AlgorithmParams,
    metrics: &mut Signal<Option<PerformanceMetrics>>,
    train_split_pct: usize,
    predictions_data: &mut Signal<Option<PredictionsData>>,
    show_predictions: &mut Signal<bool>,
) -> String {
    let start_time = web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0);

    let result = match algorithm {
        Algorithm::KMeans => run_kmeans_with_metrics(dataset, params, metrics, start_time),
        Algorithm::LogisticRegression => run_logistic_regression_with_metrics(
            dataset,
            params,
            metrics,
            start_time,
            train_split_pct,
            predictions_data,
            show_predictions,
        ),
        _ => run_algorithm(algorithm, dataset, params), // Fallback for algorithms without metrics
    };

    result
}

/// Run the selected algorithm on the dataset and return a formatted result message
fn run_algorithm(algorithm: Algorithm, dataset: &CsvDataset, params: &AlgorithmParams) -> String {
    match algorithm {
        Algorithm::KMeans => run_kmeans(dataset, params),
        Algorithm::PCA => run_pca(dataset, params),
        Algorithm::LogisticRegression => run_logistic_regression(dataset, params),
        Algorithm::DecisionTree => run_decision_tree(dataset, params),
        Algorithm::NaiveBayes => run_naive_bayes(dataset, params),
        Algorithm::StandardScaler => run_standard_scaler(dataset, params),
        Algorithm::MinMaxScaler => run_minmax_scaler(dataset, params),
    }
}

// Actual implementations - now enabled with correct trait syntax

fn run_kmeans(dataset: &CsvDataset, params: &AlgorithmParams) -> String {
    let k = params.k_clusters;
    let mut kmeans = KMeans::new(k, params.kmeans_max_iter, params.kmeans_tolerance, Some(42));

    match kmeans.fit(&dataset.features) {
        Ok(_) => {
            match kmeans.predict(&dataset.features) {
                Ok(labels) => {
                    // Count samples per cluster
                    let mut counts = vec![0; k];
                    for &label in &labels {
                        counts[label] += 1;
                    }

                    let cluster_summary: Vec<String> = counts
                        .iter()
                        .enumerate()
                        .map(|(i, &count)| format!("Cluster {}: {} samples", i, count))
                        .collect();

                    format!(
                        "✅ K-Means (k={}) completed!\n\n{}",
                        k,
                        cluster_summary.join("\n")
                    )
                }
                Err(e) => format!("❌ Prediction failed: {}", e),
            }
        }
        Err(e) => format!("❌ K-Means failed: {}", e),
    }
}

fn run_pca(dataset: &CsvDataset, params: &AlgorithmParams) -> String {
    let n_components = params.n_components.min(dataset.features.cols); // Use configured components or fewer if data has less
    let mut pca = PCA::new(n_components);

    match pca.fit(&dataset.features) {
        Ok(_) => {
            match pca.transform(&dataset.features) {
                Ok(_transformed) => {
                    // Get explained variance if available
                    let explained_text = format!(
                        "Reduced from {} to {} dimensions",
                        dataset.features.cols, n_components
                    );

                    format!("✅ PCA completed!\n\n{}", explained_text)
                }
                Err(e) => format!("❌ Transform failed: {}", e),
            }
        }
        Err(e) => format!("❌ PCA failed: {}", e),
    }
}

fn run_logistic_regression(dataset: &CsvDataset, params: &AlgorithmParams) -> String {
    // Logistic regression requires labels (targets)
    let mut model = LogisticRegression::new(
        params.learning_rate,
        params.logreg_max_iter,
        params.logreg_tolerance,
    );

    match model.fit(&dataset.features, &dataset.targets) {
        Ok(_) => {
            match model.predict(&dataset.features) {
                Ok(predictions) => {
                    // Calculate accuracy
                    let mut correct = 0;
                    for (i, &pred) in predictions.iter().enumerate() {
                        let pred_f64 = pred as f64;
                        if (pred_f64 - dataset.targets[i]).abs() < 0.5 {
                            correct += 1;
                        }
                    }
                    let accuracy = (correct as f64 / predictions.len() as f64) * 100.0;

                    // Get unique classes
                    let mut classes: Vec<f64> = dataset.targets.clone();
                    classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    classes.dedup();

                    format!(
                        "✅ Logistic Regression completed!\n\nAccuracy: {:.2}%\nClasses: {}\nSamples: {}",
                        accuracy,
                        classes.len(),
                        predictions.len()
                    )
                }
                Err(e) => format!("❌ Prediction failed: {}", e),
            }
        }
        Err(e) => format!("❌ Logistic Regression failed: {}", e),
    }
}

fn run_standard_scaler(dataset: &CsvDataset, _params: &AlgorithmParams) -> String {
    let mut scaler = StandardScaler::new();

    match scaler.fit(&dataset.features) {
        Ok(_) => match scaler.transform(&dataset.features) {
            Ok(scaled) => {
                format!(
                        "✅ StandardScaler completed!\n\nScaled {} features to μ=0, σ=1\nTransformed {} samples",
                        dataset.features.cols,
                        scaled.len()
                    )
            }
            Err(e) => format!("❌ Transform failed: {}", e),
        },
        Err(e) => format!("❌ StandardScaler failed: {}", e),
    }
}

fn run_minmax_scaler(dataset: &CsvDataset, params: &AlgorithmParams) -> String {
    let mut scaler = MinMaxScaler::new(params.scaler_min, params.scaler_max);

    match scaler.fit(&dataset.features) {
        Ok(_) => match scaler.transform(&dataset.features) {
            Ok(scaled) => {
                format!(
                        "✅ MinMaxScaler completed!\n\nScaled {} features to [{}, {}]\nTransformed {} samples",
                        dataset.features.cols,
                        params.scaler_min,
                        params.scaler_max,
                        scaled.len()
                    )
            }
            Err(e) => format!("❌ Transform failed: {}", e),
        },
        Err(e) => format!("❌ MinMaxScaler failed: {}", e),
    }
}

fn run_decision_tree(dataset: &CsvDataset, params: &AlgorithmParams) -> String {
    let mut dt = DecisionTreeClassifier::new(
        SplitCriterion::Gini,
        Some(params.dt_max_depth),
        params.dt_min_samples_split,
        params.dt_min_samples_leaf,
    );

    match dt.fit(&dataset.features, &dataset.targets) {
        Ok(_) => match dt.predict(&dataset.features) {
            Ok(predictions) => {
                // Calculate accuracy
                let correct = predictions
                    .iter()
                    .zip(dataset.targets.iter())
                    .filter(|(&pred, &actual)| pred == actual as usize)
                    .count();
                let accuracy = (correct as f64 / predictions.len() as f64) * 100.0;

                // Count classes
                let mut classes = std::collections::HashSet::new();
                for &target in &dataset.targets {
                    classes.insert(target as usize);
                }

                format!(
                    "✅ Decision Tree completed!\n\n\
                    📊 Accuracy: {:.1}%\n\
                    🌳 Tree Depth: {}\n\
                    📈 Predictions: {}\n\
                    🎯 Classes: {}\n\
                    🔧 Min Samples Split: {}\n\
                    🍃 Min Samples Leaf: {}",
                    accuracy,
                    params.dt_max_depth,
                    predictions.len(),
                    classes.len(),
                    params.dt_min_samples_split,
                    params.dt_min_samples_leaf
                )
            }
            Err(e) => format!("❌ Prediction failed: {}", e),
        },
        Err(e) => format!("❌ Decision Tree failed: {}", e),
    }
}

fn run_naive_bayes(dataset: &CsvDataset, _params: &AlgorithmParams) -> String {
    let mut nb = GaussianNB::new();

    match nb.fit(&dataset.features, &dataset.targets) {
        Ok(_) => match nb.predict(&dataset.features) {
            Ok(predictions) => {
                // Calculate accuracy
                let correct = predictions
                    .iter()
                    .zip(dataset.targets.iter())
                    .filter(|(&pred, &actual)| pred == actual as usize)
                    .count();
                let accuracy = (correct as f64 / predictions.len() as f64) * 100.0;

                // Count classes
                let mut classes = std::collections::HashSet::new();
                for &target in &dataset.targets {
                    classes.insert(target as usize);
                }

                format!(
                    "✅ Naive Bayes completed!\n\n\
                    📊 Accuracy: {:.1}%\n\
                    🎲 Model: Gaussian Naive Bayes\n\
                    📈 Predictions: {}\n\
                    🎯 Classes: {}\n\
                    📐 Assumption: Features are independent\n\
                    📊 Distribution: Gaussian (Normal)",
                    accuracy,
                    predictions.len(),
                    classes.len()
                )
            }
            Err(e) => format!("❌ Prediction failed: {}", e),
        },
        Err(e) => format!("❌ Naive Bayes failed: {}", e),
    }
}

// Algorithm runners with metrics tracking

fn run_kmeans_with_metrics(
    dataset: &CsvDataset,
    params: &AlgorithmParams,
    metrics: &mut Signal<Option<PerformanceMetrics>>,
    start_time: f64,
) -> String {
    let k = params.k_clusters;
    let max_iter = params.kmeans_max_iter;
    let tolerance = params.kmeans_tolerance;

    let mut kmeans = KMeans::new(k, max_iter, tolerance, Some(42));

    // Update metrics during training (simulated for now)
    if let Some(ref mut perf) = *metrics.write() {
        perf.status = TrainingStatus::Running;
    }

    match kmeans.fit(&dataset.features) {
        Ok(_) => {
            let elapsed = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now() - start_time)
                .unwrap_or(0.0);

            // Update final metrics
            if let Some(ref mut perf) = *metrics.write() {
                perf.current_iteration = max_iter;
                perf.status = TrainingStatus::Converged;
                perf.elapsed_time_ms = elapsed as u64;
                perf.iterations_per_second = if elapsed > 0.0 {
                    (max_iter as f64 / elapsed) * 1000.0
                } else {
                    0.0
                };
            }

            match kmeans.predict(&dataset.features) {
                Ok(labels) => {
                    let mut counts = vec![0; k];
                    for &label in &labels {
                        counts[label] += 1;
                    }

                    let cluster_summary: Vec<String> = counts
                        .iter()
                        .enumerate()
                        .map(|(i, &count)| format!("Cluster {}: {} samples", i, count))
                        .collect();

                    format!(
                        "✅ K-Means (k={}) completed in {:.2}ms!\n\n{}",
                        k,
                        elapsed,
                        cluster_summary.join("\n")
                    )
                }
                Err(e) => {
                    if let Some(ref mut perf) = *metrics.write() {
                        perf.status = TrainingStatus::Failed;
                    }
                    format!("❌ Prediction failed: {}", e)
                }
            }
        }
        Err(e) => {
            if let Some(ref mut perf) = *metrics.write() {
                perf.status = TrainingStatus::Failed;
            }
            format!("❌ K-Means failed: {}", e)
        }
    }
}

fn run_logistic_regression_with_metrics(
    dataset: &CsvDataset,
    params: &AlgorithmParams,
    metrics: &mut Signal<Option<PerformanceMetrics>>,
    start_time: f64,
    train_split_pct: usize,
    predictions_data: &mut Signal<Option<PredictionsData>>,
    show_predictions: &mut Signal<bool>,
) -> String {
    use linear_algebra::matrix::Matrix;

    let learning_rate = params.learning_rate;
    let max_iter = params.logreg_max_iter;
    let tolerance = params.logreg_tolerance;

    // Update metrics during training
    if let Some(ref mut perf) = *metrics.write() {
        perf.status = TrainingStatus::Running;
    }

    // Split data into train/test sets
    let n_samples = dataset.num_samples;
    let train_size = (n_samples * train_split_pct) / 100;

    // Extract train set
    let mut train_data: Vec<f64> = Vec::new();
    let mut train_targets: Vec<f64> = Vec::new();
    for i in 0..train_size {
        for j in 0..dataset.features.cols {
            train_data.push(*dataset.features.get(i, j).unwrap());
        }
        train_targets.push(dataset.targets[i]);
    }

    // Extract test set
    let mut test_data: Vec<f64> = Vec::new();
    let mut test_targets: Vec<f64> = Vec::new();
    for i in train_size..n_samples {
        for j in 0..dataset.features.cols {
            test_data.push(*dataset.features.get(i, j).unwrap());
        }
        test_targets.push(dataset.targets[i]);
    }

    let train_matrix = match Matrix::from_vec(train_data, train_size, dataset.features.cols) {
        Ok(m) => m,
        Err(e) => return format!("❌ Failed to create train matrix: {}", e),
    };

    let test_matrix =
        match Matrix::from_vec(test_data, n_samples - train_size, dataset.features.cols) {
            Ok(m) => m,
            Err(e) => return format!("❌ Failed to create test matrix: {}", e),
        };

    // Train model on training set
    let mut model = LogisticRegression::new(learning_rate, max_iter, tolerance);

    match model.fit(&train_matrix, &train_targets) {
        Ok(_) => {
            let elapsed = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now() - start_time)
                .unwrap_or(0.0);

            // Update final metrics
            if let Some(ref mut perf) = *metrics.write() {
                perf.current_iteration = max_iter;
                perf.status = TrainingStatus::Converged;
                perf.elapsed_time_ms = elapsed as u64;
                perf.iterations_per_second = if elapsed > 0.0 {
                    (max_iter as f64 / elapsed) * 1000.0
                } else {
                    0.0
                };
            }

            // Predict on train set
            let train_predictions = match model.predict(&train_matrix) {
                Ok(p) => p,
                Err(e) => {
                    if let Some(ref mut perf) = *metrics.write() {
                        perf.status = TrainingStatus::Failed;
                    }
                    return format!("❌ Train prediction failed: {}", e);
                }
            };

            // Predict on test set
            let test_predictions = match model.predict(&test_matrix) {
                Ok(p) => p,
                Err(e) => {
                    if let Some(ref mut perf) = *metrics.write() {
                        perf.status = TrainingStatus::Failed;
                    }
                    return format!("❌ Test prediction failed: {}", e);
                }
            };

            // Calculate accuracies
            let train_actual: Vec<usize> = train_targets.iter().map(|&x| x as usize).collect();
            let test_actual: Vec<usize> = test_targets.iter().map(|&x| x as usize).collect();

            let train_correct = train_predictions
                .iter()
                .zip(train_actual.iter())
                .filter(|(&pred, &actual)| pred == actual)
                .count();

            let test_correct = test_predictions
                .iter()
                .zip(test_actual.iter())
                .filter(|(&pred, &actual)| pred == actual)
                .count();

            let train_accuracy = (train_correct as f64 / train_predictions.len() as f64) * 100.0;
            let test_accuracy = (test_correct as f64 / test_predictions.len() as f64) * 100.0;

            // Get number of classes
            let mut classes: Vec<usize> = train_actual.clone();
            classes.extend(test_actual.clone());
            classes.sort();
            classes.dedup();
            let num_classes = classes.len();

            // Build confusion matrix (test set only)
            let mut confusion_matrix = vec![vec![0; num_classes]; num_classes];
            for (&actual, &pred) in test_actual.iter().zip(test_predictions.iter()) {
                confusion_matrix[actual][pred] += 1;
            }

            // Store predictions data
            predictions_data.set(Some(PredictionsData {
                train_actual,
                train_predicted: train_predictions.clone(),
                train_accuracy,
                test_actual,
                test_predicted: test_predictions.clone(),
                test_accuracy,
                num_classes,
                confusion_matrix,
            }));

            // Show predictions table
            show_predictions.set(true);

            format!(
                "✅ Logistic Regression completed in {:.2}ms!\n\n\
                📊 Training Accuracy: {:.1}% ({} samples)\n\
                🎯 Test Accuracy: {:.1}% ({} samples)\n\
                📈 Classes: {}\n\n\
                See detailed predictions and confusion matrix below!",
                elapsed,
                train_accuracy,
                train_predictions.len(),
                test_accuracy,
                test_predictions.len(),
                num_classes
            )
        }
        Err(e) => {
            if let Some(ref mut perf) = *metrics.write() {
                perf.status = TrainingStatus::Failed;
            }
            format!("❌ Logistic Regression training failed: {}", e)
        }
    }
}
