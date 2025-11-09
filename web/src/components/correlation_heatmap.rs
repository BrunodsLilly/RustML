use dioxus::prelude::*;
use linear_algebra::matrix::Matrix;

/// Correlation heatmap showing pairwise feature correlations
///
/// Displays:
/// - N×N grid where each cell shows correlation between two features
/// - Diverging color scale: -1 (red) → 0 (white) → +1 (blue)
/// - Interactive hover to show exact values
/// - Symmetric matrix (corr(i,j) == corr(j,i))
/// - Diagonal is 1.0 (feature perfectly correlates with itself)
///
/// # Props
/// - `correlation_matrix`: Computed correlation matrix (N×N)
/// - `feature_names`: Names of features (length N)
#[component]
pub fn CorrelationHeatmap(correlation_matrix: Matrix<f64>, feature_names: Vec<String>) -> Element {
    let n_features = feature_names.len();

    // Cell size for SVG (smaller for many features)
    let cell_size = if n_features > 10 { 40 } else { 60 };
    let font_size = if n_features > 10 { 10 } else { 12 };
    let label_width = 120;
    let total_width = label_width + cell_size * n_features;
    let total_height = 60 + cell_size * n_features; // Header + grid

    rsx! {
        div { class: "correlation-heatmap",
            div { class: "section-header",
                h3 { "🔥 Feature Correlation Matrix" }
                p { class: "subtitle", "How features relate to each other" }
            }

            // Explanation
            div { class: "info-box",
                p {
                    "Correlation ranges from -1 (perfect negative) to +1 (perfect positive). "
                    "Strong correlations (>0.7 or <-0.7) may indicate redundant features."
                }
            }

            // SVG Heatmap
            div { class: "heatmap-container",
                svg {
                    width: "{total_width}",
                    height: "{total_height}",
                    view_box: "0 0 {total_width} {total_height}",
                    xmlns: "http://www.w3.org/2000/svg",

                    // Column labels (top)
                    g { class: "column-labels",
                        for (j, name) in feature_names.iter().enumerate() {
                            text {
                                key: "col-{j}",
                                x: "{label_width + j * cell_size + cell_size / 2}",
                                y: "30",
                                text_anchor: "middle",
                                font_size: "{font_size}",
                                transform: "rotate(-45 {label_width + j * cell_size + cell_size / 2} 30)",
                                class: "feature-label",
                                "{name}"
                            }
                        }
                    }

                    // Row labels (left) and grid cells
                    g { class: "heatmap-grid",
                        transform: "translate(0, 60)",

                        for i in 0..n_features {
                            // Row label
                            text {
                                key: "row-{i}",
                                x: "{label_width - 10}",
                                y: "{i * cell_size + cell_size / 2 + 5}",
                                text_anchor: "end",
                                font_size: "{font_size}",
                                class: "feature-label",
                                "{feature_names[i]}"
                            }

                            // Grid cells for this row
                            for j in 0..n_features {
                                {
                                    let corr = correlation_matrix[(i, j)];
                                    let color = correlation_to_color(corr);
                                    let text_color = if corr.abs() > 0.5 { "#ffffff" } else { "#000000" };

                                    rsx! {
                                        g {
                                            key: "cell-{i}-{j}",
                                            class: "heatmap-cell",

                                            // Cell background
                                            rect {
                                                x: "{label_width + j * cell_size}",
                                                y: "{i * cell_size}",
                                                width: "{cell_size}",
                                                height: "{cell_size}",
                                                fill: "{color}",
                                                stroke: "#e5e7eb",
                                                stroke_width: "1",
                                            }

                                            // Cell value text
                                            text {
                                                x: "{label_width + j * cell_size + cell_size / 2}",
                                                y: "{i * cell_size + cell_size / 2 + 5}",
                                                text_anchor: "middle",
                                                font_size: "{font_size - 2}",
                                                fill: "{text_color}",
                                                class: "cell-value",
                                                "{corr:.2}"
                                            }

                                            // Interactive title for hover
                                            title {
                                                "{feature_names[i]} ↔ {feature_names[j]}: {corr:.4}"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Color scale legend
                    g { class: "color-legend",
                        transform: "translate({total_width - 100}, {total_height - 40})",

                        // Legend gradient bar
                        defs {
                            linearGradient {
                                id: "correlation-gradient",
                                x1: "0%",
                                y1: "0%",
                                x2: "100%",
                                y2: "0%",
                                stop { offset: "0%", stop_color: "#dc2626" }    // Red (-1)
                                stop { offset: "50%", stop_color: "#ffffff" }   // White (0)
                                stop { offset: "100%", stop_color: "#2563eb" }  // Blue (+1)
                            }
                        }

                        rect {
                            x: "0",
                            y: "0",
                            width: "80",
                            height: "15",
                            fill: "url(#correlation-gradient)",
                            stroke: "#9ca3af",
                            stroke_width: "1"
                        }

                        // Legend labels
                        text { x: "0", y: "30", font_size: "10", "-1" }
                        text { x: "35", y: "30", font_size: "10", "0" }
                        text { x: "70", y: "30", font_size: "10", "+1" }
                    }
                }
            }

            // Insights
            div { class: "correlation-insights",
                h4 { "Key Insights" }
                {
                    let insights = analyze_correlations(&correlation_matrix, &feature_names);
                    rsx! {
                        ul {
                            for insight in insights {
                                li { "{insight}" }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Convert correlation value to RGB color
/// -1 (red) → 0 (white) → +1 (blue)
fn correlation_to_color(corr: f64) -> String {
    let clamped = corr.clamp(-1.0, 1.0);

    if clamped >= 0.0 {
        // Positive: white → blue
        let intensity = (255.0 * (1.0 - clamped)) as u8;
        format!("rgb({}, {}, 255)", intensity, intensity)
    } else {
        // Negative: white → red
        let intensity = (255.0 * (1.0 + clamped)) as u8;
        format!("rgb(255, {}, {})", intensity, intensity)
    }
}

/// Analyze correlation matrix and generate insights
fn analyze_correlations(corr_matrix: &Matrix<f64>, feature_names: &[String]) -> Vec<String> {
    let mut insights = Vec::new();
    let n = feature_names.len();

    // Find strongest positive correlation (excluding diagonal)
    let mut max_pos_corr = 0.0;
    let mut max_pos_pair = (0, 0);

    // Find strongest negative correlation
    let mut min_neg_corr = 0.0;
    let mut min_neg_pair = (0, 0);

    for i in 0..n {
        for j in (i + 1)..n {
            let corr = corr_matrix[(i, j)];
            if corr > max_pos_corr {
                max_pos_corr = corr;
                max_pos_pair = (i, j);
            }
            if corr < min_neg_corr {
                min_neg_corr = corr;
                min_neg_pair = (i, j);
            }
        }
    }

    // Report strongest positive correlation
    if max_pos_corr > 0.7 {
        insights.push(format!(
            "⚠️ Strong positive correlation between '{}' and '{}' ({:.3}). Consider removing one.",
            feature_names[max_pos_pair.0], feature_names[max_pos_pair.1], max_pos_corr
        ));
    } else if max_pos_corr > 0.5 {
        insights.push(format!(
            "Moderate correlation between '{}' and '{}' ({:.3}).",
            feature_names[max_pos_pair.0], feature_names[max_pos_pair.1], max_pos_corr
        ));
    }

    // Report strongest negative correlation
    if min_neg_corr < -0.7 {
        insights.push(format!(
            "Strong negative correlation between '{}' and '{}' ({:.3}).",
            feature_names[min_neg_pair.0], feature_names[min_neg_pair.1], min_neg_corr
        ));
    }

    // Check for multicollinearity
    let high_corr_count = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .filter(|&(i, j)| corr_matrix[(i, j)].abs() > 0.7)
        .count();

    if high_corr_count > n / 2 {
        insights.push(
            "⚠️ Multiple high correlations detected. Model may suffer from multicollinearity."
                .to_string(),
        );
    } else if high_corr_count == 0 {
        insights.push("✓ Features are relatively independent (good for regression).".to_string());
    }

    // Average absolute correlation
    let mut sum_abs_corr = 0.0;
    let mut count = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            sum_abs_corr += corr_matrix[(i, j)].abs();
            count += 1;
        }
    }

    if count > 0 {
        let avg_abs_corr = sum_abs_corr / count as f64;
        insights.push(format!("Average correlation strength: {:.3}", avg_abs_corr));
    }

    if insights.is_empty() {
        insights.push("No significant correlations found.".to_string());
    }

    insights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_to_color() {
        // Perfect positive (blue)
        let color = correlation_to_color(1.0);
        assert_eq!(color, "rgb(0, 0, 255)");

        // Perfect negative (red)
        let color = correlation_to_color(-1.0);
        assert_eq!(color, "rgb(255, 0, 0)");

        // No correlation (white)
        let color = correlation_to_color(0.0);
        assert_eq!(color, "rgb(255, 255, 255)");
    }

    #[test]
    fn test_analyze_correlations_perfect() {
        let corr = Matrix::from_vec(vec![1.0, 1.0, 1.0, 1.0], 2, 2).unwrap();

        let names = vec!["A".to_string(), "B".to_string()];
        let insights = analyze_correlations(&corr, &names);

        // Should detect perfect correlation
        assert!(insights.iter().any(|s| s.contains("Strong positive")));
    }

    #[test]
    fn test_analyze_correlations_independent() {
        let corr = Matrix::from_vec(vec![1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();

        let names = vec!["X".to_string(), "Y".to_string()];
        let insights = analyze_correlations(&corr, &names);

        // Should report independence
        assert!(insights.iter().any(|s| s.contains("independent")));
    }
}
