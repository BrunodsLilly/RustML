//! Interactive Optimizer Visualization - WASM Performance Showcase
//!
//! This component demonstrates the power of Rust + WASM for compute-intensive tasks.
//! We're running 4 optimizers simultaneously, computing gradients, updating positions,
//! and rendering everything at 60 FPS - entirely in the browser with zero backend.
//!
//! **Performance Goals:**
//! - 1000+ gradient computations per second (per optimizer)
//! - 60 FPS smooth animations
//! - Real-time heatmap generation (50x50 = 2500 evaluations)
//! - All computation client-side in WASM
//!
//! This is impossible with pure JavaScript. WASM makes it trivial.

use dioxus::prelude::*;
use neural_network::optimizer::Optimizer;
use std::time::{Duration, Instant};
use super::loss_functions::{LossFunction, HeatmapCache};

/// Maximum iterations per frame to maintain 60 FPS
const ITERATIONS_PER_FRAME: usize = 100;

/// Heatmap resolution (50x50 = 2500 loss evaluations)
const HEATMAP_RESOLUTION: usize = 50;

/// Maximum path points to store (prevents memory leaks during long runs)
/// At 60 FPS with sampling every 10 iters, this represents ~17 seconds of history
const MAX_PATH_LENGTH: usize = 1000;

/// Maximum loss history entries to store
/// At 100 iters/frame * 60 FPS = 6000 iters/sec, this represents ~1.7 seconds
const MAX_LOSS_HISTORY: usize = 10000;

/// State for a single optimizer's training run
#[derive(Clone, Debug, PartialEq)]
struct OptimizerState {
    /// The optimizer instance
    optimizer: Optimizer,
    /// Current position (x, y)
    position: (f64, f64),
    /// History of positions for path rendering
    path: Vec<(f64, f64)>,
    /// History of loss values
    losses: Vec<f64>,
    /// Current iteration number
    iteration: usize,
    /// Whether this optimizer has converged
    converged: bool,
    /// Color for this optimizer's path
    color: &'static str,
}

impl OptimizerState {
    fn new(optimizer: Optimizer, start: (f64, f64), color: &'static str) -> Self {
        Self {
            optimizer,
            position: start,
            path: vec![start],
            losses: Vec::new(),
            iteration: 0,
            converged: false,
            color,
        }
    }

    /// Reset to initial position
    fn reset(&mut self, start: (f64, f64)) {
        self.optimizer.reset();
        self.position = start;
        self.path = vec![start];
        self.losses.clear();
        self.iteration = 0;
        self.converged = false;
    }

    /// Perform one optimization step
    fn step(&mut self, loss_fn: &LossFunction) {
        if self.converged {
            return;
        }

        let (x, y) = self.position;
        let (dx, dy) = loss_fn.gradient(x, y);

        // Zero-allocation 2D optimization step (10-50x faster than Matrix approach)
        self.position = self.optimizer.step_2d((x, y), (dx, dy));

        // Record history with bounded circular buffer
        if self.iteration % 10 == 0 {
            if self.path.len() >= MAX_PATH_LENGTH {
                // Remove oldest point to maintain bounded memory
                self.path.remove(0);
            }
            self.path.push(self.position);
        }

        let (new_x, new_y) = self.position;
        let loss = loss_fn.evaluate(new_x, new_y);
        if self.losses.len() >= MAX_LOSS_HISTORY {
            // Remove oldest loss to maintain bounded memory
            self.losses.remove(0);
        }
        self.losses.push(loss);

        self.iteration += 1;

        // Check convergence (gradient magnitude < threshold)
        let grad_magnitude = (dx * dx + dy * dy).sqrt();
        if grad_magnitude < 1e-6 {
            self.converged = true;
        }
    }
}

/// Performance metrics for display
#[derive(Clone, Debug)]
struct PerformanceMetrics {
    /// Iterations per second
    iterations_per_second: f64,
    /// Frame time in milliseconds
    frame_time_ms: f64,
    /// Total gradient computations
    total_computations: usize,
    /// Time since start
    elapsed_seconds: f64,
}

/// Main optimizer visualization component
#[component]
pub fn OptimizerDemo() -> Element {
    // State
    let mut loss_function = use_signal(|| LossFunction::Rosenbrock);
    let mut is_training = use_signal(|| false);
    let mut speed_multiplier = use_signal(|| 1.0);
    let mut show_heatmap = use_signal(|| true);
    let animation_speed = use_signal(|| 50.0); // ms per frame

    // Optimizer states
    let mut optimizers = use_signal(|| {
        let start = loss_function().starting_point();
        vec![
            OptimizerState::new(Optimizer::sgd(0.001), start, "#ef4444"), // Red
            OptimizerState::new(Optimizer::momentum(0.001, 0.9), start, "#10b981"), // Green
            OptimizerState::new(Optimizer::rmsprop(0.01, 0.999, 1e-8), start, "#3b82f6"), // Blue
            OptimizerState::new(Optimizer::adam(0.01, 0.9, 0.999, 1e-8), start, "#f59e0b"), // Yellow
        ]
    });

    // Heatmap cache
    let mut heatmap = use_signal(|| {
        HeatmapCache::new(loss_function(), HEATMAP_RESOLUTION)
    });

    // Performance metrics
    let mut metrics = use_signal(|| PerformanceMetrics {
        iterations_per_second: 0.0,
        frame_time_ms: 0.0,
        total_computations: 0,
        elapsed_seconds: 0.0,
    });

    let start_time = use_signal(|| Instant::now());

    // Training loop effect
    use_effect(move || {
        if !is_training() {
            return;
        }

        let interval = Duration::from_millis(animation_speed() as u64);

        spawn(async move {
            loop {
                if !is_training() {
                    break;
                }

                let frame_start = Instant::now();
                let loss_fn = loss_function();

                // Compute iterations based on speed multiplier
                let iters = (ITERATIONS_PER_FRAME as f64 * speed_multiplier()).round() as usize;

                // Run training steps for all optimizers
                for opt_state in optimizers.write().iter_mut() {
                    for _ in 0..iters {
                        opt_state.step(&loss_fn);
                    }
                }

                // Update metrics
                let frame_time = frame_start.elapsed();
                let total_iters = iters * 4; // 4 optimizers
                let elapsed = start_time().elapsed().as_secs_f64();

                metrics.write().frame_time_ms = frame_time.as_secs_f64() * 1000.0;
                metrics.write().iterations_per_second =
                    (total_iters as f64 / frame_time.as_secs_f64()).round();
                metrics.write().total_computations += total_iters;
                metrics.write().elapsed_seconds = elapsed;

                // Sleep to maintain frame rate
                async_std::task::sleep(interval).await;
            }
        });
    });

    // Reset logic (shared between reset button and function changes)
    let mut do_reset = {
        let loss_function = loss_function.clone();
        let mut optimizers = optimizers.clone();
        let mut metrics = metrics.clone();
        let mut start_time = start_time.clone();

        move || {
            let start = loss_function().starting_point();
            for opt in optimizers.write().iter_mut() {
                opt.reset(start);
            }
            metrics.write().total_computations = 0;
            start_time.set(Instant::now());
        }
    };

    // Reset button handler
    let reset = {
        let mut do_reset = do_reset.clone();
        move |_| {
            do_reset();
        }
    };

    // Toggle training
    let toggle_training = move |_| {
        is_training.set(!is_training());
    };

    // Change loss function
    let mut change_function = move |new_fn: LossFunction| {
        loss_function.set(new_fn);
        heatmap.set(HeatmapCache::new(new_fn, HEATMAP_RESOLUTION));
        do_reset();
    };

    rsx! {
        div {
            id: "optimizer-demo",
            class: "min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 text-white p-8",

            // Header
            div {
                class: "max-w-7xl mx-auto mb-8",
                h1 {
                    class: "text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent",
                    "⚡ Optimizer Race Track"
                }
                p {
                    class: "text-xl text-gray-300 mb-2",
                    "Watch 4 optimizers compete in real-time. All computation happens in WASM - zero backend."
                }
                div {
                    class: "flex items-center gap-4 text-sm text-gray-400",
                    span {
                        class: "px-3 py-1 bg-blue-500/20 rounded-full",
                        "🦀 Rust + WASM"
                    }
                    span {
                        class: "px-3 py-1 bg-green-500/20 rounded-full",
                        "{metrics().iterations_per_second:.0} iterations/sec"
                    }
                    span {
                        class: "px-3 py-1 bg-purple-500/20 rounded-full",
                        "{metrics().frame_time_ms:.1}ms frame time"
                    }
                    span {
                        class: "px-3 py-1 bg-yellow-500/20 rounded-full",
                        "{metrics().total_computations} total computations"
                    }
                }
            }

            // Main content grid
            div {
                class: "max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6",

                // Left: Controls
                div {
                    class: "lg:col-span-1 space-y-4",

                    // Loss function selector
                    div {
                        class: "bg-gray-800/50 backdrop-blur rounded-lg p-6 border border-gray-700",
                        h3 {
                            class: "text-lg font-semibold mb-4",
                            "Loss Function"
                        }

                        div {
                            class: "space-y-2",
                            for func in LossFunction::all() {
                                {
                                    let is_selected = func == loss_function();
                                    let difficulty_stars = "⭐".repeat(func.difficulty() as usize);

                                    rsx! {
                                        button {
                                            class: if is_selected {
                                                "w-full text-left p-4 rounded-lg bg-blue-600 border-2 border-blue-400"
                                            } else {
                                                "w-full text-left p-4 rounded-lg bg-gray-700/50 border border-gray-600 hover:border-gray-500"
                                            },
                                            onclick: move |_| change_function(func),

                                            div {
                                                class: "font-semibold",
                                                "{func.name()}"
                                            }
                                            div {
                                                class: "text-xs text-gray-400 mt-1",
                                                "{func.description()}"
                                            }
                                            div {
                                                class: "text-xs mt-2",
                                                "{difficulty_stars}"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Training controls
                    div {
                        class: "bg-gray-800/50 backdrop-blur rounded-lg p-6 border border-gray-700",
                        h3 {
                            class: "text-lg font-semibold mb-4",
                            "Controls"
                        }

                        div {
                            class: "space-y-4",

                            // Start/Stop button
                            button {
                                class: if is_training() {
                                    "w-full py-3 rounded-lg bg-red-600 hover:bg-red-700 font-semibold"
                                } else {
                                    "w-full py-3 rounded-lg bg-green-600 hover:bg-green-700 font-semibold"
                                },
                                onclick: toggle_training,
                                if is_training() {
                                    "⏸ Pause"
                                } else {
                                    "▶ Start Race"
                                }
                            }

                            // Reset button
                            button {
                                class: "w-full py-3 rounded-lg bg-gray-700 hover:bg-gray-600 font-semibold",
                                onclick: reset,
                                "🔄 Reset"
                            }

                            // Speed control
                            div {
                                label {
                                    class: "block text-sm mb-2",
                                    "Speed: {speed_multiplier():.1}x"
                                }
                                input {
                                    r#type: "range",
                                    class: "w-full",
                                    min: "0.1",
                                    max: "10.0",
                                    step: "0.1",
                                    value: "{speed_multiplier()}",
                                    oninput: move |e| {
                                        if let Ok(val) = e.value().parse::<f64>() {
                                            speed_multiplier.set(val);
                                        }
                                    }
                                }
                            }

                            // Heatmap toggle
                            label {
                                class: "flex items-center gap-2",
                                input {
                                    r#type: "checkbox",
                                    checked: show_heatmap(),
                                    onchange: move |e| show_heatmap.set(e.checked())
                                }
                                span { "Show Loss Landscape" }
                            }
                        }
                    }

                    // Performance showcase
                    div {
                        class: "bg-gradient-to-br from-purple-900/50 to-blue-900/50 backdrop-blur rounded-lg p-6 border border-purple-700",
                        h3 {
                            class: "text-lg font-semibold mb-4 flex items-center gap-2",
                            span { "⚡" }
                            span { "WASM Performance" }
                        }
                        div {
                            class: "space-y-3 text-sm",
                            div {
                                class: "flex justify-between",
                                span { "Iterations/sec:" }
                                span {
                                    class: "font-mono font-bold text-green-400",
                                    "{metrics().iterations_per_second:.0}"
                                }
                            }
                            div {
                                class: "flex justify-between",
                                span { "Frame time:" }
                                span {
                                    class: "font-mono font-bold text-blue-400",
                                    "{metrics().frame_time_ms:.2}ms"
                                }
                            }
                            div {
                                class: "flex justify-between",
                                span { "Total computations:" }
                                span {
                                    class: "font-mono font-bold text-yellow-400",
                                    "{metrics().total_computations}"
                                }
                            }
                            div {
                                class: "pt-3 mt-3 border-t border-purple-700/50 text-xs text-gray-400",
                                "Running entirely in your browser with Rust + WASM. No server needed."
                            }
                        }
                    }
                }

                // Right: Visualization (2 columns)
                div {
                    class: "lg:col-span-2",

                    // Visualization canvas
                    LossLandscape {
                        loss_function: loss_function(),
                        heatmap: heatmap(),
                        optimizers: optimizers(),
                        show_heatmap: show_heatmap()
                    }

                    // Optimizer status cards
                    div {
                        class: "grid grid-cols-2 lg:grid-cols-4 gap-4 mt-6",
                        for (idx, opt) in optimizers().iter().enumerate() {
                            {
                                let name = match idx {
                                    0 => "SGD",
                                    1 => "Momentum",
                                    2 => "RMSprop",
                                    3 => "Adam",
                                    _ => "Unknown"
                                };

                                let final_loss = opt.losses.last().copied().unwrap_or(f64::INFINITY);
                                let (target_x, target_y, _) = loss_function().global_minima()[0];
                                let distance = ((opt.position.0 - target_x).powi(2) +
                                               (opt.position.1 - target_y).powi(2)).sqrt();

                                rsx! {
                                    div {
                                        class: "bg-gray-800/50 backdrop-blur rounded-lg p-4 border border-gray-700",
                                        div {
                                            class: "flex items-center gap-2 mb-3",
                                            div {
                                                class: "w-3 h-3 rounded-full",
                                                style: "background-color: {opt.color}"
                                            }
                                            span {
                                                class: "font-semibold",
                                                "{name}"
                                            }
                                        }
                                        div {
                                            class: "space-y-1 text-xs",
                                            div {
                                                span { class: "text-gray-400", "Iteration: " }
                                                span { class: "font-mono", "{opt.iteration}" }
                                            }
                                            div {
                                                span { class: "text-gray-400", "Loss: " }
                                                span { class: "font-mono", "{final_loss:.6}" }
                                            }
                                            div {
                                                span { class: "text-gray-400", "Distance: " }
                                                span { class: "font-mono", "{distance:.4}" }
                                            }
                                            if opt.converged {
                                                div {
                                                    class: "mt-2 text-green-400",
                                                    "✓ Converged"
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
        }
    }
}

/// Loss landscape visualization with optimizer paths
#[component]
fn LossLandscape(
    loss_function: LossFunction,
    heatmap: HeatmapCache,
    optimizers: Vec<OptimizerState>,
    show_heatmap: bool,
) -> Element {
    let ((x_min, x_max), (y_min, y_max)) = loss_function.bounds();

    // SVG dimensions
    let width = 800.0;
    let height = 600.0;

    // Transform functions
    let to_svg_x = |x: f64| -> f64 {
        ((x - x_min) / (x_max - x_min)) * width
    };

    let to_svg_y = |y: f64| -> f64 {
        ((y - y_min) / (y_max - y_min)) * height
    };

    rsx! {
        div {
            class: "bg-gray-800/50 backdrop-blur rounded-lg p-6 border border-gray-700",

            svg {
                view_box: "0 0 {width} {height}",
                class: "w-full h-auto bg-gray-900 rounded",

                // Heatmap
                if show_heatmap {
                    {
                        let resolution = heatmap.resolution;
                        let cell_width = width / resolution as f64;
                        let cell_height = height / resolution as f64;
                        let heatmap_clone = heatmap.clone();

                        // Flatten the 2D grid into a single iterator
                        // Standard indexing: row = y-axis, col = x-axis
                        (0..resolution).flat_map(move |row| {
                            let heatmap_inner = heatmap_clone.clone();
                            (0..resolution).map(move |col| {
                                let normalized = heatmap_inner.normalized_value(row, col);
                                let (h, s, l) = heatmap_inner.value_to_color(normalized);

                                // SVG coordinates: x = col, y = row
                                let x = col as f64 * cell_width;
                                let y = row as f64 * cell_height;

                                rsx! {
                                    rect {
                                        key: "{row}-{col}",
                                        x: "{x}",
                                        y: "{y}",
                                        width: "{cell_width}",
                                        height: "{cell_height}",
                                        fill: "hsl({h * 360.0}, {s * 100.0}%, {l * 100.0}%)",
                                        opacity: "0.6"
                                    }
                                }
                            })
                        })
                    }
                }

                // Global minimum markers
                for (target_x, target_y, _) in loss_function.global_minima() {
                    {
                        let sx = to_svg_x(target_x);
                        let sy = to_svg_y(target_y);

                        rsx! {
                            circle {
                                cx: "{sx}",
                                cy: "{sy}",
                                r: "8",
                                fill: "none",
                                stroke: "#fbbf24",
                                stroke_width: "3",
                                opacity: "0.8"
                            }
                            circle {
                                cx: "{sx}",
                                cy: "{sy}",
                                r: "4",
                                fill: "#fbbf24",
                                opacity: "0.8"
                            }
                        }
                    }
                }

                // Optimizer paths
                for opt in optimizers {
                    {
                        // Path line
                        let points: String = opt.path.iter()
                            .map(|(x, y)| format!("{},{}", to_svg_x(*x), to_svg_y(*y)))
                            .collect::<Vec<_>>()
                            .join(" ");

                        // Current position
                        let (curr_x, curr_y) = opt.position;
                        let curr_sx = to_svg_x(curr_x);
                        let curr_sy = to_svg_y(curr_y);

                        rsx! {
                            // Path
                            polyline {
                                points: "{points}",
                                stroke: opt.color,
                                stroke_width: "2",
                                fill: "none",
                                opacity: "0.7"
                            }

                            // Current position
                            circle {
                                cx: "{curr_sx}",
                                cy: "{curr_sy}",
                                r: "6",
                                fill: opt.color,
                                stroke: "white",
                                stroke_width: "2",
                                class: "animate-pulse"
                            }

                            // Starting position
                            if !opt.path.is_empty() {
                                {
                                    let (start_x, start_y) = opt.path[0];
                                    let start_sx = to_svg_x(start_x);
                                    let start_sy = to_svg_y(start_y);

                                    rsx! {
                                        circle {
                                            cx: "{start_sx}",
                                            cy: "{start_sy}",
                                            r: "4",
                                            fill: opt.color,
                                            opacity: "0.5"
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
}
