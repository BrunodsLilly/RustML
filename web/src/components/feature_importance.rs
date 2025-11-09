use dioxus::prelude::*;

/// Feature importance visualization showing relative contribution of each feature
///
/// Displays:
/// - Horizontal bar chart with standardized coefficients
/// - Color coding: positive (blue), negative (red)
/// - Sortable by absolute value or alphabetical
/// - Interactive tooltips with exact values
///
/// # Props
/// - `weights`: Raw regression weights
/// - `feature_names`: Names corresponding to weights
/// - `feature_variances`: Variance of each feature (for standardization)
#[component]
pub fn FeatureImportanceChart(
    weights: Vec<f64>,
    feature_names: Vec<String>,
    feature_variances: Vec<f64>,
) -> Element {
    let mut sort_by_importance = use_signal(|| true);

    // Calculate standardized coefficients (weight * std_dev)
    let standardized: Vec<(String, f64, f64)> = feature_names
        .iter()
        .zip(weights.iter())
        .zip(feature_variances.iter())
        .map(|((name, &weight), &variance)| {
            let std_dev = variance.sqrt().max(1e-10);
            let standardized_coef = weight * std_dev;
            (name.clone(), weight, standardized_coef)
        })
        .collect();

    // Sort features
    let mut sorted = standardized.clone();
    if sort_by_importance() {
        sorted.sort_by(|a, b| {
            b.2.abs()
                .partial_cmp(&a.2.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
    }

    // Find max absolute value for scaling
    let max_abs_importance = sorted
        .iter()
        .map(|(_, _, std)| std.abs())
        .fold(0.0f64, f64::max)
        .max(1e-10);

    rsx! {
        div { class: "feature-importance-chart",
            div { class: "section-header",
                h3 { "⭐ Feature Importance" }
                p { class: "subtitle", "Relative contribution of each feature (standardized)" }
            }

            // Sort controls
            div { class: "chart-controls",
                label { "Sort by:" }
                div { class: "button-group",
                    button {
                        class: if sort_by_importance() { "active" } else { "" },
                        onclick: move |_| sort_by_importance.set(true),
                        "Importance"
                    }
                    button {
                        class: if !sort_by_importance() { "active" } else { "" },
                        onclick: move |_| sort_by_importance.set(false),
                        "Name (A-Z)"
                    }
                }
            }

            // Info box
            div { class: "info-box",
                p {
                    "Standardized coefficients show the change in the target variable (in standard deviations) "
                    "for a one standard deviation change in the feature, holding other features constant."
                }
            }

            // Bar chart
            div { class: "importance-bars",
                for (i, (name, raw_weight, std_coef)) in sorted.iter().enumerate() {
                    {
                        let bar_width = (std_coef.abs() / max_abs_importance * 100.0).min(100.0);
                        let is_positive = *std_coef > 0.0;
                        let is_top_feature = i == 0 && sort_by_importance();

                        rsx! {
                            div {
                                key: "{name}",
                                class: if is_top_feature { "importance-row highlight" } else { "importance-row" },

                                // Feature name
                                div { class: "importance-label",
                                    span { class: "feature-name", "{name}" }
                                    if is_top_feature {
                                        span { class: "badge top", "most important" }
                                    }
                                }

                                // Bar visualization
                                div { class: "importance-bar-container",
                                    div {
                                        class: if is_positive { "importance-bar positive" } else { "importance-bar negative" },
                                        style: "width: {bar_width}%",
                                        title: "Standardized: {std_coef:.4}, Raw: {raw_weight:.4}"
                                    }
                                }

                                // Numeric value
                                div { class: "importance-value",
                                    span {
                                        class: if is_positive { "positive" } else { "negative" },
                                        "{std_coef:.4}"
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Legend
            div { class: "chart-legend",
                div { class: "legend-row",
                    div { class: "legend-item",
                        span { class: "legend-color positive" }
                        span { "Positive influence (increases target)" }
                    }
                    div { class: "legend-item",
                        span { class: "legend-color negative" }
                        span { "Negative influence (decreases target)" }
                    }
                }
            }

            // Statistics summary
            div { class: "importance-stats",
                div { class: "stat-card",
                    h4 { "Top Feature" }
                    p { class: "stat-value", "{sorted[0].0}" }
                    p { class: "stat-detail", "Importance: {sorted[0].2.abs():.4}" }
                }
                div { class: "stat-card",
                    h4 { "Average Importance" }
                    {
                        let avg = sorted.iter().map(|(_, _, s)| s.abs()).sum::<f64>() / sorted.len() as f64;
                        rsx! {
                            p { class: "stat-value", "{avg:.4}" }
                        }
                    }
                }
                div { class: "stat-card",
                    h4 { "Direction" }
                    {
                        let pos_count = sorted.iter().filter(|(_, _, s)| *s > 0.0).count();
                        let neg_count = sorted.len() - pos_count;
                        rsx! {
                            p { class: "stat-value", "+{pos_count} / -{neg_count}" }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standardized_coefficient_calculation() {
        // Feature with weight=2.0 and variance=4.0 (std=2.0)
        // Standardized coefficient = 2.0 * 2.0 = 4.0
        let weight: f64 = 2.0;
        let variance: f64 = 4.0;
        let std_dev = variance.sqrt();
        let standardized = weight * std_dev;
        assert_eq!(standardized, 4.0);
    }

    #[test]
    fn test_zero_variance_handling() {
        // Should not crash with zero variance
        let variance: f64 = 0.0;
        let std_dev = variance.sqrt().max(1e-10);
        assert!(std_dev > 0.0);
    }
}
