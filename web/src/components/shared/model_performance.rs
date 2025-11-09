use dioxus::prelude::*;

/// Training status
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingStatus {
    NotStarted,
    Running,
    Converged,
    MaxIterationsReached,
    Diverged,
    Failed,
}

impl TrainingStatus {
    pub fn icon(&self) -> &'static str {
        match self {
            TrainingStatus::NotStarted => "⏸️",
            TrainingStatus::Running => "▶️",
            TrainingStatus::Converged => "✅",
            TrainingStatus::MaxIterationsReached => "🔄",
            TrainingStatus::Diverged => "⚠️",
            TrainingStatus::Failed => "❌",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            TrainingStatus::NotStarted => "Not Started",
            TrainingStatus::Running => "Training...",
            TrainingStatus::Converged => "Converged",
            TrainingStatus::MaxIterationsReached => "Max Iterations",
            TrainingStatus::Diverged => "Diverged",
            TrainingStatus::Failed => "Failed",
        }
    }

    pub fn color(&self) -> &'static str {
        match self {
            TrainingStatus::NotStarted => "#95a5a6",
            TrainingStatus::Running => "#3498db",
            TrainingStatus::Converged => "#27ae60",
            TrainingStatus::MaxIterationsReached => "#f39c12",
            TrainingStatus::Diverged => "#e67e22",
            TrainingStatus::Failed => "#e74c3c",
        }
    }
}

/// Model performance metrics
#[derive(Clone, Debug, PartialEq)]
pub struct PerformanceMetrics {
    pub current_iteration: usize,
    pub max_iterations: usize,
    pub current_loss: f64,
    pub best_loss: f64,
    pub loss_history: Vec<f64>,
    pub status: TrainingStatus,
    pub elapsed_time_ms: u64,
    pub iterations_per_second: f64,
}

impl PerformanceMetrics {
    pub fn new(max_iterations: usize) -> Self {
        Self {
            current_iteration: 0,
            max_iterations,
            current_loss: std::f64::INFINITY,
            best_loss: std::f64::INFINITY,
            loss_history: Vec::new(),
            status: TrainingStatus::NotStarted,
            elapsed_time_ms: 0,
            iterations_per_second: 0.0,
        }
    }

    pub fn progress_percent(&self) -> f64 {
        if self.max_iterations == 0 {
            0.0
        } else {
            (self.current_iteration as f64 / self.max_iterations as f64 * 100.0).min(100.0)
        }
    }

    pub fn improvement_percent(&self) -> f64 {
        if !self.loss_history.is_empty() && self.loss_history[0].is_finite() && self.loss_history[0] > 0.0 {
            let initial = self.loss_history[0];
            ((initial - self.best_loss) / initial * 100.0).max(0.0)
        } else {
            0.0
        }
    }

    pub fn estimated_time_remaining_ms(&self) -> u64 {
        if self.iterations_per_second > 0.0 && self.current_iteration < self.max_iterations {
            let remaining = self.max_iterations - self.current_iteration;
            ((remaining as f64 / self.iterations_per_second) * 1000.0) as u64
        } else {
            0
        }
    }
}

/// Props for ModelPerformanceCard component
#[derive(Props, Clone, PartialEq)]
pub struct ModelPerformanceCardProps {
    /// Performance metrics
    metrics: PerformanceMetrics,
    /// Model name/type
    #[props(default = "Model")]
    model_name: &'static str,
    /// Show loss chart
    #[props(default = true)]
    show_loss_chart: bool,
    /// Show detailed metrics
    #[props(default = true)]
    show_details: bool,
}

/// Model performance card with training progress visualization
///
/// # Example
/// ```rust
/// let mut metrics = use_signal(|| PerformanceMetrics::new(1000));
///
/// rsx! {
///     ModelPerformanceCard {
///         metrics: metrics(),
///         model_name: "Linear Regression",
///         show_loss_chart: true,
///     }
/// }
/// ```
#[component]
pub fn ModelPerformanceCard(props: ModelPerformanceCardProps) -> Element {
    let progress = props.metrics.progress_percent();
    let improvement = props.metrics.improvement_percent();
    let eta = format_time(props.metrics.estimated_time_remaining_ms());

    rsx! {
        div { class: "model-performance-card",
            // Header with status
            div { class: "perf-header",
                h3 {
                    span { class: "status-icon", "{props.metrics.status.icon()}" }
                    "{props.model_name}"
                }
                span {
                    class: "status-badge",
                    style: "background-color: {props.metrics.status.color()}",
                    "{props.metrics.status.label()}"
                }
            }

            // Progress bar
            div { class: "progress-section",
                div { class: "progress-bar-container",
                    div {
                        class: "progress-bar",
                        style: "width: {progress}%; background: linear-gradient(90deg, #6c5ce7 0%, #a29bfe {progress}%)",
                    }
                }
                div { class: "progress-stats",
                    span { "Iteration {props.metrics.current_iteration}/{props.metrics.max_iterations}" }
                    span { "{progress:.1}% Complete" }
                }
            }

            // Key metrics grid
            div { class: "metrics-grid",
                div { class: "metric-card loss",
                    span { class: "metric-label", "Current Loss" }
                    span { class: "metric-value",
                        if props.metrics.current_loss.is_finite() {
                            "{props.metrics.current_loss:.6}"
                        } else {
                            "N/A"
                        }
                    }
                }

                div { class: "metric-card best",
                    span { class: "metric-label", "Best Loss" }
                    span { class: "metric-value",
                        if props.metrics.best_loss.is_finite() {
                            "{props.metrics.best_loss:.6}"
                        } else {
                            "N/A"
                        }
                    }
                }

                div { class: "metric-card improvement",
                    span { class: "metric-label", "Improvement" }
                    span { class: "metric-value",
                        if improvement > 0.0 {
                            "↓ {improvement:.2}%"
                        } else {
                            "N/A"
                        }
                    }
                }

                div { class: "metric-card speed",
                    span { class: "metric-label", "Speed" }
                    span { class: "metric-value", "{props.metrics.iterations_per_second:.1} it/s" }
                }
            }

            // Loss chart
            if props.show_loss_chart && !props.metrics.loss_history.is_empty() {
                div { class: "loss-chart-section",
                    h4 { "📉 Loss Over Time" }
                    LossChart {
                        history: props.metrics.loss_history.clone(),
                        current_iteration: props.metrics.current_iteration,
                    }
                }
            }

            // Detailed metrics
            if props.show_details {
                div { class: "details-section",
                    div { class: "detail-row",
                        span { class: "detail-label", "⏱️ Elapsed Time:" }
                        span { class: "detail-value", "{format_time(props.metrics.elapsed_time_ms)}" }
                    }
                    div { class: "detail-row",
                        span { class: "detail-label", "⏳ Estimated Remaining:" }
                        span { class: "detail-value", "{eta}" }
                    }
                    div { class: "detail-row",
                        span { class: "detail-label", "📊 Samples Processed:" }
                        span { class: "detail-value", "{props.metrics.current_iteration}" }
                    }
                }
            }
        }
    }
}

fn format_time(ms: u64) -> String {
    let seconds = ms / 1000;
    let minutes = seconds / 60;
    let hours = minutes / 60;

    if hours > 0 {
        format!("{}h {}m", hours, minutes % 60)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds % 60)
    } else {
        format!("{}s", seconds)
    }
}

/// Props for LossChart component
#[derive(Props, Clone, PartialEq)]
pub struct LossChartProps {
    history: Vec<f64>,
    current_iteration: usize,
}

/// SVG loss chart visualization
#[component]
pub fn LossChart(props: LossChartProps) -> Element {
    const WIDTH: f64 = 600.0;
    const HEIGHT: f64 = 200.0;
    const PADDING: f64 = 40.0;

    let chart_width = WIDTH - 2.0 * PADDING;
    let chart_height = HEIGHT - 2.0 * PADDING;

    if props.history.is_empty() {
        return rsx! {
            div { class: "loss-chart-empty", "No data yet" }
        };
    }

    // Filter out non-finite values and get valid data
    let valid_data: Vec<(usize, f64)> = props.history
        .iter()
        .enumerate()
        .filter(|(_, &loss)| loss.is_finite())
        .map(|(i, &loss)| (i, loss))
        .collect();

    if valid_data.is_empty() {
        return rsx! {
            div { class: "loss-chart-empty", "No valid data" }
        };
    }

    // Calculate bounds
    let min_loss = valid_data.iter().map(|(_, loss)| loss).fold(f64::INFINITY, |a, &b| a.min(b));
    let max_loss = valid_data.iter().map(|(_, loss)| loss).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let loss_range = (max_loss - min_loss).max(1e-10);

    // Build path
    let mut path_data = String::from("M ");
    for (i, (iter, loss)) in valid_data.iter().enumerate() {
        let x = PADDING + (*iter as f64 / props.history.len() as f64) * chart_width;
        let y = PADDING + chart_height - ((loss - min_loss) / loss_range * chart_height);

        if i == 0 {
            path_data.push_str(&format!("{} {}", x, y));
        } else {
            path_data.push_str(&format!(" L {} {}", x, y));
        }
    }

    rsx! {
        svg {
            class: "loss-chart",
            width: "{WIDTH}",
            height: "{HEIGHT}",
            view_box: "0 0 {WIDTH} {HEIGHT}",

            // Grid lines
            for i in 0..5 {
                {
                    let y = PADDING + (i as f64 / 4.0) * chart_height;
                    rsx! {
                        line {
                            x1: "{PADDING}",
                            y1: "{y}",
                            x2: "{PADDING + chart_width}",
                            y2: "{y}",
                            stroke: "#e9ecef",
                            stroke_width: "1",
                        }
                    }
                }
            }

            // Loss line
            path {
                d: "{path_data}",
                fill: "none",
                stroke: "url(#lossGradient)",
                stroke_width: "3",
                stroke_linejoin: "round",
                stroke_linecap: "round",
            }

            // Current position indicator
            {
                let x = PADDING + (props.current_iteration as f64 / props.history.len() as f64) * chart_width;
                rsx! {
                    circle {
                        cx: "{x}",
                        cy: "{PADDING + chart_height / 2.0}",
                        r: "5",
                        fill: "#ff6b6b",
                        class: "pulse"
                    }
                }
            }

            // Gradient definition
            defs {
                linearGradient {
                    id: "lossGradient",
                    x1: "0%",
                    y1: "0%",
                    x2: "100%",
                    y2: "0%",

                    stop { offset: "0%", stop_color: "#6c5ce7" }
                    stop { offset: "100%", stop_color: "#a29bfe" }
                }
            }

            // Axes labels
            g { class: "axes-labels",
                text { x: "{WIDTH / 2.0}", y: "{HEIGHT - 10.0}", font_size: "12", fill: "#666", text_anchor: "middle",
                    "Iteration"
                }
                text { x: "15", y: "{HEIGHT / 2.0}", font_size: "12", fill: "#666", text_anchor: "middle", transform: "rotate(-90 15 {HEIGHT / 2.0})",
                    "Loss"
                }
                text { x: "{PADDING}", y: "{PADDING - 10.0}", font_size: "10", fill: "#666",
                    "Max: {max_loss:.3}"
                }
                text { x: "{PADDING}", y: "{HEIGHT - PADDING + 20.0}", font_size: "10", fill: "#666",
                    "Min: {min_loss:.3}"
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_status() {
        assert_eq!(TrainingStatus::Converged.icon(), "✅");
        assert_eq!(TrainingStatus::Running.label(), "Training...");
    }

    #[test]
    fn test_progress_calculation() {
        let metrics = PerformanceMetrics {
            current_iteration: 50,
            max_iterations: 100,
            current_loss: 0.5,
            best_loss: 0.4,
            loss_history: vec![1.0, 0.8, 0.6, 0.5],
            status: TrainingStatus::Running,
            elapsed_time_ms: 5000,
            iterations_per_second: 10.0,
        };

        assert_eq!(metrics.progress_percent(), 50.0);
    }

    #[test]
    fn test_improvement_calculation() {
        let metrics = PerformanceMetrics {
            current_iteration: 10,
            max_iterations: 100,
            current_loss: 0.5,
            best_loss: 0.3,
            loss_history: vec![1.0, 0.8, 0.6, 0.5, 0.4, 0.3],
            status: TrainingStatus::Running,
            elapsed_time_ms: 1000,
            iterations_per_second: 10.0,
        };

        assert_eq!(metrics.improvement_percent(), 70.0); // (1.0 - 0.3) / 1.0 * 100
    }

    #[test]
    fn test_format_time() {
        assert_eq!(format_time(1000), "1s");
        assert_eq!(format_time(65000), "1m 5s");
        assert_eq!(format_time(3665000), "1h 1m");
    }

    #[test]
    fn test_eta_calculation() {
        let metrics = PerformanceMetrics {
            current_iteration: 50,
            max_iterations: 100,
            current_loss: 0.5,
            best_loss: 0.4,
            loss_history: vec![],
            status: TrainingStatus::Running,
            elapsed_time_ms: 5000,
            iterations_per_second: 10.0, // 10 iterations per second
        };

        // Remaining: 50 iterations at 10 it/s = 5 seconds = 5000 ms
        assert_eq!(metrics.estimated_time_remaining_ms(), 5000);
    }
}
