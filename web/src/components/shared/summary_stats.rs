use dioxus::prelude::*;

/// Statistical summary for a numeric column
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnStats {
    pub name: String,
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub q1: f64, // 25th percentile
    pub q3: f64, // 75th percentile
    pub missing: usize,
}

impl ColumnStats {
    /// Compute statistics from a column of numeric values
    pub fn from_values(name: String, values: &[f64]) -> Self {
        let count = values.len();

        if count == 0 {
            return Self::empty(name);
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = values.iter().sum::<f64>() / count as f64;
        let median = percentile(&sorted, 50.0);
        let q1 = percentile(&sorted, 25.0);
        let q3 = percentile(&sorted, 75.0);
        let min = sorted.first().copied().unwrap_or(0.0);
        let max = sorted.last().copied().unwrap_or(0.0);

        // Standard deviation
        let variance = values
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        Self {
            name,
            count,
            mean,
            median,
            std_dev,
            min,
            max,
            q1,
            q3,
            missing: 0,
        }
    }

    /// Create empty stats for non-numeric columns
    pub fn empty(name: String) -> Self {
        Self {
            name,
            count: 0,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            q1: 0.0,
            q3: 0.0,
            missing: 0,
        }
    }

    /// Get interquartile range (IQR)
    pub fn iqr(&self) -> f64 {
        self.q3 - self.q1
    }

    /// Get coefficient of variation (CV)
    pub fn cv(&self) -> f64 {
        if self.mean.abs() < 1e-10 {
            0.0
        } else {
            (self.std_dev / self.mean.abs()) * 100.0
        }
    }
}

/// Compute percentile from sorted values
fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    let n = sorted_values.len();
    let index = (p / 100.0) * (n - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let fraction = index - lower as f64;

    if lower == upper {
        sorted_values[lower]
    } else {
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    }
}

/// Histogram bin for distribution visualization
#[derive(Clone, Debug, PartialEq)]
pub struct HistogramBin {
    pub start: f64,
    pub end: f64,
    pub count: usize,
    pub frequency: f64, // Normalized [0, 1]
}

/// Compute histogram bins from values
pub fn compute_histogram(values: &[f64], num_bins: usize) -> Vec<HistogramBin> {
    if values.is_empty() || num_bins == 0 {
        return Vec::new();
    }

    let min = values
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .unwrap_or(0.0);
    let max = values
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .unwrap_or(0.0);

    let range = max - min;
    if range < 1e-10 {
        // All values are the same
        return vec![HistogramBin {
            start: min,
            end: max,
            count: values.len(),
            frequency: 1.0,
        }];
    }

    let bin_width = range / num_bins as f64;
    let mut bins: Vec<HistogramBin> = (0..num_bins)
        .map(|i| {
            let start = min + i as f64 * bin_width;
            let end = if i == num_bins - 1 {
                max
            } else {
                start + bin_width
            };
            HistogramBin {
                start,
                end,
                count: 0,
                frequency: 0.0,
            }
        })
        .collect();

    // Count values in each bin
    for &value in values {
        let bin_idx = ((value - min) / bin_width)
            .floor()
            .min((num_bins - 1) as f64) as usize;
        bins[bin_idx].count += 1;
    }

    // Normalize frequencies
    let max_count = bins.iter().map(|b| b.count).max().unwrap_or(1);
    for bin in &mut bins {
        bin.frequency = bin.count as f64 / max_count as f64;
    }

    bins
}

/// Props for SummaryStats component
#[derive(Props, Clone, PartialEq)]
pub struct SummaryStatsProps {
    /// Statistics for each numeric column
    stats: Vec<ColumnStats>,
    /// Raw data for histogram generation (column-major: [column][row])
    #[props(default)]
    column_data: Option<Vec<Vec<f64>>>,
    /// Number of histogram bins
    #[props(default = 20)]
    num_bins: usize,
    /// Show distribution histograms
    #[props(default = true)]
    show_distributions: bool,
}

/// Summary statistics component with distribution histograms
///
/// # Example
/// ```rust
/// let stats = vec![
///     ColumnStats::from_values("Age".to_string(), &age_data),
///     ColumnStats::from_values("Income".to_string(), &income_data),
/// ];
///
/// let column_data = vec![age_data, income_data];
///
/// rsx! {
///     SummaryStats {
///         stats: stats,
///         column_data: Some(column_data),
///         show_distributions: true,
///     }
/// }
/// ```
#[component]
pub fn SummaryStats(props: SummaryStatsProps) -> Element {
    rsx! {
        div { class: "summary-stats-container",
            h3 { class: "summary-stats-title", "📊 Statistical Summary" }

            // Statistics cards grid
            div { class: "stats-grid",
                for (idx, stat) in props.stats.iter().enumerate() {
                    div { class: "stat-card", key: "{idx}",
                        // Column name header
                        div { class: "stat-header",
                            h4 { "{stat.name}" }
                            span { class: "stat-count", "{stat.count} values" }
                        }

                        // Key metrics
                        div { class: "stat-metrics",
                            div { class: "metric",
                                span { class: "metric-label", "Mean" }
                                span { class: "metric-value", "{stat.mean:.3}" }
                            }
                            div { class: "metric",
                                span { class: "metric-label", "Median" }
                                span { class: "metric-value", "{stat.median:.3}" }
                            }
                            div { class: "metric",
                                span { class: "metric-label", "Std Dev" }
                                span { class: "metric-value", "{stat.std_dev:.3}" }
                            }
                            div { class: "metric",
                                span { class: "metric-label", "Min" }
                                span { class: "metric-value", "{stat.min:.3}" }
                            }
                            div { class: "metric",
                                span { class: "metric-label", "Max" }
                                span { class: "metric-value", "{stat.max:.3}" }
                            }
                            div { class: "metric",
                                span { class: "metric-label", "Q1 (25%)" }
                                span { class: "metric-value", "{stat.q1:.3}" }
                            }
                            div { class: "metric",
                                span { class: "metric-label", "Q3 (75%)" }
                                span { class: "metric-value", "{stat.q3:.3}" }
                            }
                            div { class: "metric",
                                span { class: "metric-label", "IQR" }
                                span { class: "metric-value", "{stat.iqr():.3}" }
                            }
                            div { class: "metric",
                                span { class: "metric-label", "CV (%)" }
                                span { class: "metric-value", "{stat.cv():.2}" }
                            }
                        }

                        // Distribution histogram
                        if props.show_distributions {
                            if let Some(ref data) = props.column_data {
                                if let Some(column_values) = data.get(idx) {
                                    {
                                        let histogram = compute_histogram(column_values, props.num_bins);
                                        rsx! {
                                            div { class: "distribution-section",
                                                h5 { "Distribution" }
                                                DistributionHistogram {
                                                    bins: histogram,
                                                    stats: stat.clone(),
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

            // Overall dataset summary
            if !props.stats.is_empty() {
                div { class: "dataset-summary",
                    h4 { "Dataset Overview" }
                    p {
                        "Total columns: {props.stats.len()} | "
                        "Numeric features: {props.stats.iter().filter(|s| s.count > 0).count()}"
                    }
                }
            }
        }
    }
}

/// Props for DistributionHistogram component
#[derive(Props, Clone, PartialEq)]
pub struct DistributionHistogramProps {
    bins: Vec<HistogramBin>,
    stats: ColumnStats,
}

/// SVG histogram visualization
#[component]
pub fn DistributionHistogram(props: DistributionHistogramProps) -> Element {
    const WIDTH: f64 = 400.0;
    const HEIGHT: f64 = 150.0;
    const PADDING: f64 = 30.0;

    let chart_width = WIDTH - 2.0 * PADDING;
    let chart_height = HEIGHT - 2.0 * PADDING;

    if props.bins.is_empty() {
        return rsx! {
            div { class: "histogram-empty", "No data to display" }
        };
    }

    let bin_width = chart_width / props.bins.len() as f64;

    rsx! {
        svg {
            class: "distribution-histogram",
            width: "{WIDTH}",
            height: "{HEIGHT}",
            view_box: "0 0 {WIDTH} {HEIGHT}",

            // Histogram bars
            for (i, bin) in props.bins.iter().enumerate() {
                {
                    let x = PADDING + i as f64 * bin_width;
                    let bar_height = bin.frequency * chart_height;
                    let y = PADDING + chart_height - bar_height;

                    rsx! {
                        rect {
                            x: "{x}",
                            y: "{y}",
                            width: "{bin_width * 0.9}",
                            height: "{bar_height}",
                            fill: "url(#histogramGradient)",
                            class: "histogram-bar",

                            title { "{bin.count} values in [{bin.start:.2}, {bin.end:.2})" }
                        }
                    }
                }
            }

            // Mean line
            {
                let mean_x = PADDING + ((props.stats.mean - props.stats.min) / (props.stats.max - props.stats.min)) * chart_width;
                rsx! {
                    line {
                        x1: "{mean_x}",
                        y1: "{PADDING}",
                        x2: "{mean_x}",
                        y2: "{PADDING + chart_height}",
                        stroke: "#ff6b6b",
                        stroke_width: "2",
                        stroke_dasharray: "5,5",

                        title { "Mean: {props.stats.mean:.3}" }
                    }
                }
            }

            // Median line
            {
                let median_x = PADDING + ((props.stats.median - props.stats.min) / (props.stats.max - props.stats.min)) * chart_width;
                rsx! {
                    line {
                        x1: "{median_x}",
                        y1: "{PADDING}",
                        x2: "{median_x}",
                        y2: "{PADDING + chart_height}",
                        stroke: "#51cf66",
                        stroke_width: "2",
                        stroke_dasharray: "3,3",

                        title { "Median: {props.stats.median:.3}" }
                    }
                }
            }

            // Gradient definition
            defs {
                linearGradient {
                    id: "histogramGradient",
                    x1: "0%",
                    y1: "100%",
                    x2: "0%",
                    y2: "0%",

                    stop { offset: "0%", stop_color: "#6c5ce7", stop_opacity: "0.6" }
                    stop { offset: "100%", stop_color: "#a29bfe", stop_opacity: "0.9" }
                }
            }

            // Legend
            g { transform: "translate({PADDING}, {HEIGHT - 15.0})",
                text { x: "0", y: "0", font_size: "10", fill: "#666",
                    "Min: {props.stats.min:.2}"
                }
                text { x: "{chart_width}", y: "0", font_size: "10", fill: "#666", text_anchor: "end",
                    "Max: {props.stats.max:.2}"
                }
            }
        }

        // Legend for mean/median lines
        div { class: "histogram-legend",
            span { class: "legend-item",
                span { class: "legend-line mean-line" }
                "Mean"
            }
            span { class: "legend-item",
                span { class: "legend-line median-line" }
                "Median"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&values, 0.0), 1.0);
        assert_eq!(percentile(&values, 50.0), 3.0);
        assert_eq!(percentile(&values, 100.0), 5.0);
    }

    #[test]
    fn test_column_stats_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = ColumnStats::from_values("test".to_string(), &values);

        assert_eq!(stats.count, 5);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_histogram_binning() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let bins = compute_histogram(&values, 5);

        assert_eq!(bins.len(), 5);
        assert!(bins.iter().all(|b| b.count > 0));
    }

    #[test]
    fn test_iqr_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = ColumnStats::from_values("test".to_string(), &values);

        let iqr = stats.iqr();
        assert!(iqr > 0.0);
    }

    #[test]
    fn test_cv_calculation() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let stats = ColumnStats::from_values("test".to_string(), &values);

        let cv = stats.cv();
        assert!(cv > 0.0 && cv < 100.0);
    }

    #[test]
    fn test_empty_stats() {
        let stats = ColumnStats::empty("test".to_string());
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
    }
}
