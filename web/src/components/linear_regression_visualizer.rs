use dioxus::prelude::*;
use linear_algebra::{matrix::Matrix, statistics::correlation_matrix};
use linear_regression::LinearRegressor;
use loader::CsvDataset;

use crate::components::{CoefficientDisplay, CorrelationHeatmap, FeatureImportanceChart};

/// Unified visualization component for linear regression analysis
///
/// Provides tabbed interface showing:
/// 1. Model Coefficients - Learned weights and bias
/// 2. Feature Importance - Standardized coefficients
/// 3. Correlations - Feature relationship heatmap
///
/// # Props
/// - `model`: Trained LinearRegressor with weights and bias
/// - `dataset`: CsvDataset with features and target
#[component]
pub fn LinearRegressionVisualizer(
    model: ReadOnlySignal<LinearRegressor>,
    dataset: ReadOnlySignal<CsvDataset>,
) -> Element {
    let mut active_tab = use_signal(|| "coefficients");

    let model_val = model();
    let dataset_val = dataset();

    // Compute feature variances for standardization
    let feature_variances: Vec<f64> = (0..dataset_val.features.cols)
        .map(|j| {
            let col = dataset_val.features.col(j).expect("Valid column index");
            let mean = col.data.iter().sum::<f64>() / col.data.len() as f64;
            let variance = col.data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / (col.data.len() - 1) as f64;
            variance
        })
        .collect();

    // Compute correlation matrix
    let corr_matrix = correlation_matrix(&dataset_val.features);

    rsx! {
        div { class: "linear-regression-visualizer",
            // Header
            div { class: "visualizer-header",
                h2 { "📊 Model Analysis & Insights" }
                p { class: "subtitle",
                    "Explore how your model works: {dataset_val.num_samples} samples, {dataset_val.features.cols} features"
                }
            }

            // Tab navigation
            div { class: "tab-navigation",
                button {
                    class: if active_tab() == "coefficients" { "tab-button active" } else { "tab-button" },
                    onclick: move |_| active_tab.set("coefficients"),
                    "📋 Coefficients"
                }
                button {
                    class: if active_tab() == "importance" { "tab-button active" } else { "tab-button" },
                    onclick: move |_| active_tab.set("importance"),
                    "⭐ Importance"
                }
                button {
                    class: if active_tab() == "correlations" { "tab-button active" } else { "tab-button" },
                    onclick: move |_| active_tab.set("correlations"),
                    "🔥 Correlations"
                }
            }

            // Tab content
            div { class: "tab-content",
                // Coefficients Tab
                if active_tab() == "coefficients" {
                    div { class: "tab-panel",
                        CoefficientDisplay {
                            weights: model_val.weights.data.clone(),
                            feature_names: dataset_val.feature_names.clone(),
                            bias: model_val.bias,
                        }
                    }
                }

                // Importance Tab
                if active_tab() == "importance" {
                    div { class: "tab-panel",
                        FeatureImportanceChart {
                            weights: model_val.weights.data.clone(),
                            feature_names: dataset_val.feature_names.clone(),
                            feature_variances: feature_variances.clone(),
                        }
                    }
                }

                // Correlations Tab
                if active_tab() == "correlations" {
                    div { class: "tab-panel",
                        CorrelationHeatmap {
                            correlation_matrix: corr_matrix.clone(),
                            feature_names: dataset_val.feature_names.clone(),
                        }
                    }
                }
            }

            // Model Performance Summary (always visible)
            div { class: "performance-summary",
                h3 { "Model Performance" }
                div { class: "summary-grid",
                    div { class: "summary-card",
                        h4 { "Final Cost" }
                        p { class: "big-number",
                            "{model_val.training_history.last().unwrap_or(&0.0):.6}"
                        }
                    }
                    div { class: "summary-card",
                        h4 { "Training Iterations" }
                        p { class: "big-number",
                            "{model_val.training_history.len()}"
                        }
                    }
                    div { class: "summary-card",
                        h4 { "Cost Reduction" }
                        {
                            let reduction = if model_val.training_history.len() > 1 {
                                let start = model_val.training_history[0];
                                let end = model_val.training_history.last().copied().unwrap_or(start);
                                ((start - end) / start * 100.0).max(0.0).min(99.99)
                            } else {
                                0.0
                            };
                            rsx! {
                                p { class: "big-number positive",
                                    "{reduction:.1}%"
                                }
                            }
                        }
                    }
                }
            }

            // Quick tips based on data
            div { class: "tips-panel",
                h4 { "💡 Tips" }
                {
                    let tips = generate_tips(&dataset_val, &corr_matrix);
                    rsx! {
                        ul {
                            for tip in tips {
                                li { "{tip}" }
                            }
                        }
                    }
                }
            }

            // Export options
            div { class: "export-panel",
                h4 { "Export" }
                div { class: "export-buttons",
                    button {
                        class: "export-btn",
                        onclick: move |_| {
                            // TODO: Implement JSON export
                        },
                        "💾 Save Coefficients (JSON)"
                    }
                    button {
                        class: "export-btn",
                        onclick: move |_| {
                            // TODO: Implement visualization export
                        },
                        "📸 Export Visualization (PNG)"
                    }
                }
            }
        }
    }
}

/// Generate contextual tips based on dataset and model characteristics
fn generate_tips(dataset: &CsvDataset, corr_matrix: &Matrix<f64>) -> Vec<String> {
    let mut tips = Vec::new();
    let n_features = dataset.features.cols;

    // Check for high correlations
    let high_corr_count = (0..n_features)
        .flat_map(|i| (i + 1..n_features).map(move |j| (i, j)))
        .filter(|&(i, j)| corr_matrix[(i, j)].abs() > 0.7)
        .count();

    if high_corr_count > 0 {
        tips.push(format!(
            "Found {} pairs with high correlation (>0.7). Check the Correlations tab.",
            high_corr_count
        ));
    }

    // Check feature count
    if n_features > 10 {
        tips.push(
            "With many features, consider using feature importance to identify the most relevant ones."
                .to_string(),
        );
    }

    // Check sample size
    let samples_per_feature = dataset.num_samples as f64 / n_features as f64;
    if samples_per_feature < 10.0 {
        tips.push(format!(
            "You have {:.1} samples per feature. More data (10+ samples/feature) improves reliability.",
            samples_per_feature
        ));
    }

    if tips.is_empty() {
        tips.push(
            "Your dataset looks good! Explore the tabs to understand your model.".to_string(),
        );
    }

    tips
}

#[cfg(test)]
mod tests {
    use super::*;
    use linear_algebra::vectors::Vector;

    #[test]
    fn test_generate_tips_high_correlation() {
        // Create dataset with 2 perfectly correlated features
        let features = Matrix::from_vec(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], 3, 2).unwrap();

        let dataset = CsvDataset {
            features,
            targets: vec![1.0, 2.0, 3.0],
            feature_names: vec!["A".to_string(), "B".to_string()],
            num_samples: 3,
        };

        let corr = correlation_matrix(&dataset.features);
        let tips = generate_tips(&dataset, &corr);

        // Should warn about high correlation
        assert!(tips.iter().any(|t| t.contains("high correlation")));
    }

    #[test]
    fn test_generate_tips_many_features() {
        // Create dataset with many features
        let mut data = vec![];
        for _i in 0..3 {
            for _j in 0..15 {
                data.push(1.0);
            }
        }

        let features = Matrix::from_vec(data, 3, 15).unwrap();
        let dataset = CsvDataset {
            features,
            targets: vec![1.0, 2.0, 3.0],
            feature_names: (0..15).map(|i| format!("F{}", i)).collect(),
            num_samples: 3,
        };

        let corr = correlation_matrix(&dataset.features);
        let tips = generate_tips(&dataset, &corr);

        // Should mention feature importance
        assert!(tips.iter().any(|t| t.contains("feature importance")));
    }

    #[test]
    fn test_generate_tips_small_sample() {
        // Create dataset with few samples relative to features
        let features = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();

        let dataset = CsvDataset {
            features,
            targets: vec![1.0, 2.0],
            feature_names: vec!["X".to_string(), "Y".to_string(), "Z".to_string()],
            num_samples: 2,
        };

        let corr = correlation_matrix(&dataset.features);
        let tips = generate_tips(&dataset, &corr);

        // Should warn about sample size
        assert!(tips.iter().any(|t| t.contains("samples per feature")));
    }
}
