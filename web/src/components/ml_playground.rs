//! ML Playground - Interactive ML algorithms with CSV upload
//!
//! This component provides a comprehensive interface for testing all ML algorithms
//! with user-uploaded CSV data.

use clustering::kmeans::KMeans;
use dimensionality_reduction::pca::PCA;
use dioxus::prelude::*;
use loader::csv_loader::CsvDataset;
use ml_traits::clustering::Clusterer;
use ml_traits::preprocessing::Transformer;
use ml_traits::supervised::SupervisedModel;
use ml_traits::unsupervised::UnsupervisedModel;
use preprocessing::scalers::{MinMaxScaler, StandardScaler};
use supervised::logistic_regression::LogisticRegression;

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
                                            let result = run_algorithm_with_metrics(
                                                *selected_algorithm.read(),
                                                dataset,
                                                &algorithm_params.read(),
                                                &mut performance_metrics
                                            );
                                            result_message.set(result);
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

    fn to_algorithm_type(&self) -> AlgorithmType {
        match self {
            Algorithm::KMeans => AlgorithmType::KMeans,
            Algorithm::PCA => AlgorithmType::PCA,
            Algorithm::LogisticRegression => AlgorithmType::LogisticRegression,
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
            scaler_min: 0.0,
            scaler_max: 1.0,
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

/// Run algorithm with performance metrics tracking
fn run_algorithm_with_metrics(
    algorithm: Algorithm,
    dataset: &CsvDataset,
    params: &AlgorithmParams,
    metrics: &mut Signal<Option<PerformanceMetrics>>,
) -> String {
    let start_time = web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0);

    let result = match algorithm {
        Algorithm::KMeans => run_kmeans_with_metrics(dataset, params, metrics, start_time),
        Algorithm::LogisticRegression => {
            run_logistic_regression_with_metrics(dataset, params, metrics, start_time)
        }
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
) -> String {
    let learning_rate = params.learning_rate;
    let max_iter = params.logreg_max_iter;
    let tolerance = params.logreg_tolerance;

    let mut model = LogisticRegression::new(learning_rate, max_iter, tolerance);

    // Update metrics during training
    if let Some(ref mut perf) = *metrics.write() {
        perf.status = TrainingStatus::Running;
    }

    match model.fit(&dataset.features, &dataset.targets) {
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

            match model.predict(&dataset.features) {
                Ok(predictions) => {
                    let mut correct = 0;
                    for (i, &pred) in predictions.iter().enumerate() {
                        let pred_f64 = pred as f64;
                        if (pred_f64 - dataset.targets[i]).abs() < 0.5 {
                            correct += 1;
                        }
                    }
                    let accuracy = (correct as f64 / predictions.len() as f64) * 100.0;

                    let mut classes: Vec<f64> = dataset.targets.clone();
                    classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    classes.dedup();

                    format!(
                        "✅ Logistic Regression completed in {:.2}ms!\n\nAccuracy: {:.2}%\nClasses: {}\nSamples: {}",
                        elapsed,
                        accuracy,
                        classes.len(),
                        predictions.len()
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
            format!("❌ Logistic Regression failed: {}", e)
        }
    }
}
