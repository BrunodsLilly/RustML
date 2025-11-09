use dioxus::prelude::*;

/// Display learned coefficients (weights and bias) from a linear regression model
///
/// Shows:
/// - Feature names with their corresponding weights
/// - Bias term (intercept)
/// - Highlights largest magnitude coefficients
/// - Model equation in readable format
///
/// # Props
/// - `weights`: Vec of learned weights (one per feature)
/// - `feature_names`: Names of features corresponding to weights
/// - `bias`: Learned bias/intercept term
#[component]
pub fn CoefficientDisplay(weights: Vec<f64>, feature_names: Vec<String>, bias: f64) -> Element {
    // Find max absolute weight for highlighting
    let max_abs_weight = weights.iter().map(|w| w.abs()).fold(0.0f64, f64::max);

    // Generate model equation string
    let equation = build_equation(&weights, &feature_names, bias);

    rsx! {
        div { class: "coefficient-display",
            div { class: "section-header",
                h3 { "📊 Model Coefficients" }
                p { class: "subtitle", "Learned weights for each feature" }
            }

            // Coefficients table
            div { class: "coefficients-table",
                table {
                    thead {
                        tr {
                            th { "Feature" }
                            th { class: "text-right", "Weight" }
                            th { class: "text-right", "Magnitude" }
                            th { "Impact" }
                        }
                    }
                    tbody {
                        // Feature weights
                        for (i, (name, &weight)) in feature_names.iter().zip(weights.iter()).enumerate() {
                            {
                                let abs_weight = weight.abs();
                                let is_largest = abs_weight == max_abs_weight && max_abs_weight > 0.0;
                                let impact_bar_width = if max_abs_weight > 0.0 {
                                    (abs_weight / max_abs_weight * 100.0).min(100.0)
                                } else {
                                    0.0
                                };

                                rsx! {
                                    tr {
                                        key: "{i}",
                                        class: if is_largest { "highlight-row" } else { "" },
                                        td { class: "feature-name",
                                            span { class: "feature-label", "{name}" }
                                            if is_largest {
                                                span { class: "badge", "strongest" }
                                            }
                                        }
                                        td { class: "text-right mono",
                                            span {
                                                class: if weight > 0.0 { "positive" } else { "negative" },
                                                "{weight:.6}"
                                            }
                                        }
                                        td { class: "text-right mono", "{abs_weight:.6}" }
                                        td { class: "impact-bar-cell",
                                            div { class: "impact-bar-container",
                                                div {
                                                    class: if weight > 0.0 { "impact-bar positive" } else { "impact-bar negative" },
                                                    style: "width: {impact_bar_width}%"
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Bias row (separator)
                        tr { class: "bias-row",
                            td { class: "feature-name",
                                span { class: "feature-label", "Bias (Intercept)" }
                            }
                            td { class: "text-right mono",
                                span {
                                    class: if bias > 0.0 { "positive" } else { "negative" },
                                    "{bias:.6}"
                                }
                            }
                            td { class: "text-right", "—" }
                            td { "—" }
                        }
                    }
                }
            }

            // Model equation
            div { class: "model-equation",
                h4 { "Model Equation" }
                div { class: "equation-display",
                    code { "{equation}" }
                    button {
                        class: "copy-btn",
                        title: "Copy equation to clipboard",
                        onclick: move |_| {
                            // In real implementation, would use clipboard API
                            // For now, just show visual feedback
                        },
                        "📋 Copy"
                    }
                }
                p { class: "equation-hint",
                    "This is the learned linear function: y = f(x₁, x₂, ...)"
                }
            }

            // Quick stats
            div { class: "coefficient-stats",
                div { class: "stat-item",
                    span { class: "stat-label", "Features:" }
                    span { class: "stat-value", "{weights.len()}" }
                }
                div { class: "stat-item",
                    span { class: "stat-label", "Positive weights:" }
                    span { class: "stat-value", "{weights.iter().filter(|&&w| w > 0.0).count()}" }
                }
                div { class: "stat-item",
                    span { class: "stat-label", "Negative weights:" }
                    span { class: "stat-value", "{weights.iter().filter(|&&w| w < 0.0).count()}" }
                }
            }
        }
    }
}

/// Build readable equation string from coefficients
fn build_equation(weights: &[f64], feature_names: &[String], bias: f64) -> String {
    if weights.is_empty() {
        return format!("y = {:.4}", bias);
    }

    let mut eq = String::from("y = ");

    for (i, (name, &weight)) in feature_names.iter().zip(weights.iter()).enumerate() {
        if i > 0 {
            if weight >= 0.0 {
                eq.push_str(" + ");
            } else {
                eq.push_str(" - ");
            }
            eq.push_str(&format!("{:.4}·{}", weight.abs(), name));
        } else {
            eq.push_str(&format!("{:.4}·{}", weight, name));
        }
    }

    // Add bias term
    if bias >= 0.0 {
        eq.push_str(&format!(" + {:.4}", bias));
    } else {
        eq.push_str(&format!(" - {:.4}", bias.abs()));
    }

    eq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_equation_single_feature() {
        let weights = vec![2.5];
        let names = vec!["x".to_string()];
        let bias = 1.0;

        let eq = build_equation(&weights, &names, bias);
        assert_eq!(eq, "y = 2.5000·x + 1.0000");
    }

    #[test]
    fn test_build_equation_multiple_features() {
        let weights = vec![1.5, -2.3, 0.8];
        let names = vec!["x1".to_string(), "x2".to_string(), "x3".to_string()];
        let bias = -0.5;

        let eq = build_equation(&weights, &names, bias);
        assert_eq!(eq, "y = 1.5000·x1 - 2.3000·x2 + 0.8000·x3 - 0.5000");
    }

    #[test]
    fn test_build_equation_zero_weights() {
        let weights = vec![0.0, 0.0];
        let names = vec!["a".to_string(), "b".to_string()];
        let bias = 5.0;

        let eq = build_equation(&weights, &names, bias);
        assert!(eq.contains("0.0000·a"));
        assert!(eq.contains("0.0000·b"));
        assert!(eq.contains("+ 5.0000"));
    }

    #[test]
    fn test_build_equation_empty() {
        let weights = vec![];
        let names = vec![];
        let bias = 3.14;

        let eq = build_equation(&weights, &names, bias);
        assert_eq!(eq, "y = 3.1400");
    }
}
