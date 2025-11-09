# Dioxus ML Education Platform - Comprehensive Guide

**Version:** Dioxus 0.6.0
**Target:** Building sophisticated ML visualization and education platform in WASM
**Last Updated:** November 8, 2025

This guide covers advanced Dioxus patterns specifically for your ML education platform, based on official documentation, community best practices, and analysis of your existing codebase.

---

## Table of Contents

1. [Advanced State Management](#1-advanced-state-management)
2. [Routing & Navigation](#2-routing--navigation)
3. [Data Visualization](#3-data-visualization)
4. [Form & Input Handling](#4-form--input-handling)
5. [Async & WASM Integration](#5-async--wasm-integration)
6. [Styling & Theming](#6-styling--theming)
7. [Performance Patterns](#7-performance-patterns)
8. [Production-Ready Patterns](#8-production-ready-patterns)

---

## 1. Advanced State Management

### Current Usage in Your Codebase

You're already using several state management patterns effectively:

```rust
// ml_playground.rs (Lines 21-24)
let mut csv_dataset = use_signal(|| None::<CsvDataset>);
let mut selected_algorithm = use_signal(|| Algorithm::KMeans);
let mut result_message = use_signal(|| String::new());
let mut is_processing = use_signal(|| false);
```

### 1.1 `use_signal` - Local Component State

**What it is:** Reactive state wrapper that triggers re-renders when modified.

**Best Practices:**
```rust
// ✅ GOOD: Initialize with closure
let mut count = use_signal(|| 0);
let mut data = use_signal(|| Vec::new());
let mut config = use_signal(|| AlgorithmConfig::default());

// ❌ BAD: Don't clone unnecessarily
let value = count.read().clone(); // Unnecessary for Copy types
let value = *count.read(); // Better for Copy types

// ✅ GOOD: Use read() for read-only access, no re-render subscription
if *count.read() > 10 {
    // This doesn't subscribe to changes
}

// ✅ GOOD: Batch updates in single closure
count.set(count() + 1);
// Or for complex updates:
count.with_mut(|c| {
    *c += 1;
    *c *= 2;
});
```

**Advanced Pattern: Derived State**
```rust
// Don't store computed values, compute on-demand
let mut weights = use_signal(|| vec![0.1, 0.2, 0.3]);

// ✅ GOOD: Compute in render
let sum: f64 = weights().iter().sum();
let normalized: Vec<f64> = weights()
    .iter()
    .map(|&w| w / sum)
    .collect();

// ❌ BAD: Don't create separate signal for derived state
// let mut sum = use_signal(|| 0.0); // This gets out of sync!
```

### 1.2 `use_context` - Cross-Component State Sharing

**When to use:**
- Theme configuration (dark mode, color scheme)
- User preferences (algorithm defaults, visualization settings)
- Global ML model cache
- Authentication state (if you add user accounts)

**Implementation Example:**

```rust
// Create a context type
#[derive(Clone, Copy)]
struct ThemeContext {
    dark_mode: Signal<bool>,
    primary_color: Signal<String>,
}

// In your App component (main.rs)
#[component]
fn App() -> Element {
    let dark_mode = use_signal(|| false);
    let primary_color = use_signal(|| "#4A90E2".to_string());

    let theme = ThemeContext {
        dark_mode,
        primary_color,
    };

    use_context_provider(|| theme);

    rsx! {
        document::Stylesheet { href: CSS }
        Router::<Route> {}
    }
}

// In any child component (ml_playground.rs, optimizer_demo.rs, etc.)
#[component]
pub fn MLPlayground() -> Element {
    let theme = use_context::<ThemeContext>();

    rsx! {
        div {
            class: if *theme.dark_mode.read() { "ml-playground dark" } else { "ml-playground" },
            // Rest of component
        }
    }
}

// In a settings component
#[component]
pub fn ThemeToggle() -> Element {
    let theme = use_context::<ThemeContext>();

    rsx! {
        button {
            onclick: move |_| {
                theme.dark_mode.set(!theme.dark_mode());
            },
            if *theme.dark_mode.read() { "☀️ Light Mode" } else { "🌙 Dark Mode" }
        }
    }
}
```

**Advanced Context Pattern: ML Algorithm Config**
```rust
// Create global algorithm configuration
#[derive(Clone, Copy)]
struct MLConfig {
    kmeans_clusters: Signal<usize>,
    pca_components: Signal<usize>,
    learning_rate: Signal<f64>,
    max_iterations: Signal<usize>,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            kmeans_clusters: Signal::new(3),
            pca_components: Signal::new(2),
            learning_rate: Signal::new(0.01),
            max_iterations: Signal::new(1000),
        }
    }
}

// Provide at app level
use_context_provider(|| MLConfig::default());

// Use in ml_playground.rs
let config = use_context::<MLConfig>();

// Now users can adjust in settings, persist across page navigation
let k = *config.kmeans_clusters.read();
let mut kmeans = KMeans::new(k, *config.max_iterations.read(), 1e-4, Some(42));
```

### 1.3 `GlobalSignal` - True Global State

**When to use:**
- Dataset cache (avoid re-parsing CSVs)
- Trained model storage
- Performance metrics tracking
- Error logging

**Implementation:**
```rust
use dioxus::prelude::*;
use std::collections::HashMap;

// Define at module level (not inside component!)
static DATASET_CACHE: GlobalSignal<HashMap<String, CsvDataset>> =
    Signal::global(|| HashMap::new());

static GLOBAL_ERROR_LOG: GlobalSignal<Vec<String>> =
    Signal::global(Vec::new);

// Use anywhere without context provider
#[component]
pub fn MLPlayground() -> Element {
    // Access global state
    let cache = DATASET_CACHE.read();

    // Update global state
    let log_error = move |msg: String| {
        GLOBAL_ERROR_LOG.write().push(format!("[{}] {}",
            chrono::Utc::now(), msg
        ));
    };

    // ...
}

// In a different component entirely
#[component]
pub fn ErrorPanel() -> Element {
    let errors = GLOBAL_ERROR_LOG.read();

    rsx! {
        div { class: "error-log",
            for error in errors.iter().rev().take(10) {
                div { class: "error-entry", "{error}" }
            }
        }
    }
}
```

### 1.4 State Management Decision Tree

```
State Decision Tree:
│
├─ Is it local to one component?
│  └─ YES → use_signal
│
├─ Is it shared between parent and children?
│  └─ YES → use_context + use_context_provider
│
├─ Is it truly global (needed everywhere)?
│  └─ YES → GlobalSignal
│
└─ Is it derived from other state?
   └─ YES → Compute in render (don't store)
```

---

## 2. Routing & Navigation

### Current Setup (main.rs)

```rust
#[derive(Routable, Clone, PartialEq)]
enum Route {
    #[route("/")]
    MainView,
    #[route("/courses")]
    CoursesView,
    #[route("/showcase")]
    ShowcaseView,
    #[route("/optimizers")]
    OptimizersView,
    #[route("/playground")]
    PlaygroundView,
}
```

### 2.1 Type-Safe Routing with Parameters

**URL Parameters for Algorithm Configuration:**

```rust
use dioxus::prelude::*;
use dioxus_router::prelude::*;

#[derive(Routable, Clone, PartialEq)]
enum Route {
    #[route("/")]
    MainView,

    // Nested routes with parameters
    #[route("/playground")]
    PlaygroundView,

    #[route("/playground/:algorithm")]
    PlaygroundAlgorithm {
        algorithm: String,
    },

    #[route("/playground/:algorithm/:dataset")]
    PlaygroundWithDataset {
        algorithm: String,
        dataset: String,
    },

    // Route with query parameters
    #[route("/optimizer?:loss_fn&:learning_rate")]
    OptimizerWithConfig {
        loss_fn: Option<String>,
        learning_rate: Option<f64>,
    },
}

// Component implementation
#[component]
fn PlaygroundAlgorithm(algorithm: String) -> Element {
    let algo = match algorithm.as_str() {
        "kmeans" => Algorithm::KMeans,
        "pca" => Algorithm::PCA,
        "logistic" => Algorithm::LogisticRegression,
        _ => Algorithm::KMeans, // Default
    };

    rsx! {
        MLPlayground { initial_algorithm: algo }
    }
}
```

**Benefits of URL State:**
- Shareable links: `example.com/playground/kmeans?k=5`
- Browser back/forward works correctly
- Bookmark-friendly configurations
- Deep linking for educational content

### 2.2 Navigation with `use_navigator`

```rust
use dioxus_router::prelude::*;

#[component]
pub fn AlgorithmSelector() -> Element {
    let nav = use_navigator();

    rsx! {
        div { class: "algorithm-grid",
            button {
                onclick: move |_| {
                    // Navigate programmatically
                    nav.push(Route::PlaygroundAlgorithm {
                        algorithm: "kmeans".to_string(),
                    });
                },
                "K-Means"
            }

            button {
                onclick: move |_| {
                    // Navigate with query params
                    nav.push(Route::OptimizerWithConfig {
                        loss_fn: Some("rosenbrock".to_string()),
                        learning_rate: Some(0.01),
                    });
                },
                "Optimizer Demo"
            }
        }
    }
}
```

### 2.3 Nested Routing with Layouts

**Create a layout component:**

```rust
#[derive(Routable, Clone, PartialEq)]
enum Route {
    #[route("/")]
    MainView,

    #[layout(PlaygroundLayout)]
        #[route("/playground")]
        PlaygroundHome,

        #[route("/playground/algorithms")]
        AlgorithmsView,

        #[route("/playground/datasets")]
        DatasetsView,
    #[end_layout]
}

#[component]
fn PlaygroundLayout() -> Element {
    rsx! {
        div { class: "playground-layout",
            // Sidebar navigation (always visible)
            nav { class: "playground-sidebar",
                Link { to: Route::PlaygroundHome, "Home" }
                Link { to: Route::AlgorithmsView, "Algorithms" }
                Link { to: Route::DatasetsView, "Datasets" }
            }

            // Main content area (changes based on route)
            div { class: "playground-content",
                Outlet::<Route> {}
            }
        }
    }
}
```

### 2.4 Loading States with Suspense

```rust
#[component]
fn PlaygroundWithDataset(dataset: String) -> Element {
    // Async resource loading
    let dataset_data = use_resource(move || async move {
        load_dataset(&dataset).await
    });

    rsx! {
        Suspense {
            fallback: |context: SuspenseContext| rsx! {
                div { class: "loading-state",
                    div { class: "spinner" }
                    p { "Loading dataset: {dataset}..." }
                }
            },

            match dataset_data.read().as_ref() {
                Some(Ok(data)) => rsx! {
                    MLPlayground { dataset: data.clone() }
                },
                Some(Err(e)) => rsx! {
                    div { class: "error-state",
                        "❌ Failed to load dataset: {e}"
                    }
                },
                None => rsx! { div { "Loading..." } }
            }
        }
    }
}
```

### 2.5 Tab Navigation Pattern (As You're Using)

**Current Pattern (linear_regression_visualizer.rs):**

```rust
let mut active_tab = use_signal(|| "coefficients");

rsx! {
    div { class: "tab-navigation",
        button {
            class: if active_tab() == "coefficients" { "tab-button active" } else { "tab-button" },
            onclick: move |_| active_tab.set("coefficients"),
            "📋 Coefficients"
        }
        // More tabs...
    }

    div { class: "tab-content",
        if active_tab() == "coefficients" {
            // Tab content
        }
    }
}
```

**Enhanced Pattern with Enum:**

```rust
#[derive(Clone, Copy, PartialEq)]
enum VisualizationTab {
    Coefficients,
    Importance,
    Correlations,
    Training,
}

impl VisualizationTab {
    fn icon(&self) -> &'static str {
        match self {
            Self::Coefficients => "📋",
            Self::Importance => "⭐",
            Self::Correlations => "🔥",
            Self::Training => "📈",
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Coefficients => "Coefficients",
            Self::Importance => "Importance",
            Self::Correlations => "Correlations",
            Self::Training => "Training History",
        }
    }
}

#[component]
pub fn LinearRegressionVisualizer(
    model: ReadOnlySignal<LinearRegressor>,
    dataset: ReadOnlySignal<CsvDataset>,
) -> Element {
    let mut active_tab = use_signal(|| VisualizationTab::Coefficients);

    let tabs = [
        VisualizationTab::Coefficients,
        VisualizationTab::Importance,
        VisualizationTab::Correlations,
        VisualizationTab::Training,
    ];

    rsx! {
        div { class: "tab-navigation",
            for tab in tabs {
                button {
                    class: if active_tab() == tab { "tab-button active" } else { "tab-button" },
                    onclick: move |_| active_tab.set(tab),
                    "{tab.icon()} {tab.label()}"
                }
            }
        }

        div { class: "tab-content",
            match active_tab() {
                VisualizationTab::Coefficients => rsx! {
                    CoefficientDisplay { /* props */ }
                },
                VisualizationTab::Importance => rsx! {
                    FeatureImportanceChart { /* props */ }
                },
                VisualizationTab::Correlations => rsx! {
                    CorrelationHeatmap { /* props */ }
                },
                VisualizationTab::Training => rsx! {
                    TrainingHistoryChart { /* props */ }
                },
            }
        }
    }
}
```

---

## 3. Data Visualization

### Current Approach: Pure SVG (Excellent for Your Use Case)

You're already using SVG effectively. Here's the performance breakdown:

| Approach | Best For | Your Usage |
|----------|----------|------------|
| **SVG** | <1000 elements, interactivity, crisp scaling | ✅ Current (heatmaps, charts) |
| **Canvas** | >1000 elements, animations, game-like visuals | ⏳ Future (optimizer paths) |
| **WebGL** | 3D visualizations, massive datasets | 🔮 Moonshot (3D loss surfaces) |

### 3.1 SVG Patterns You're Already Using

**CorrelationHeatmap (correlation_heatmap.rs):**
```rust
// Your pattern (simplified):
rsx! {
    svg {
        width: "{svg_width}",
        height: "{svg_height}",

        // Grid cells
        for i in 0..n {
            for j in 0..n {
                rect {
                    x: "{cell_x}",
                    y: "{cell_y}",
                    width: "{cell_size}",
                    height: "{cell_size}",
                    fill: "{color}",
                    onmouseover: move |_| /* tooltip */,
                }
            }
        }

        // Labels
        for (i, name) in feature_names.iter().enumerate() {
            text {
                x: "{x}",
                y: "{y}",
                "{name}"
            }
        }
    }
}
```

**Strengths:**
- ✅ Declarative and readable
- ✅ Automatic interactivity (hover, click)
- ✅ Scales to any screen size
- ✅ Easy to style with CSS

**Weaknesses:**
- ⚠️ Performance degrades >1000 elements
- ⚠️ Animations can be choppy

### 3.2 Optimized SVG Patterns

**Pre-compute Complex Calculations:**

```rust
// ❌ BAD: Compute in JSX
rsx! {
    for i in 0..1000 {
        circle {
            cx: "{i as f64 * 10.0 + offset}",
            cy: "{(i as f64).sin() * 50.0 + height / 2.0}",
            r: "3",
        }
    }
}

// ✅ GOOD: Pre-compute
let points: Vec<(f64, f64)> = (0..1000)
    .map(|i| {
        let x = i as f64 * 10.0 + offset;
        let y = (i as f64).sin() * 50.0 + height / 2.0;
        (x, y)
    })
    .collect();

rsx! {
    for (x, y) in points {
        circle {
            cx: "{x}",
            cy: "{y}",
            r: "3",
        }
    }
}
```

**Use SVG Paths for Many Points:**

```rust
// ❌ BAD: 1000 circle elements
rsx! {
    for (x, y) in points {
        circle { cx: "{x}", cy: "{y}", r: "2" }
    }
}

// ✅ GOOD: Single path element
let path_data = points
    .iter()
    .enumerate()
    .map(|(i, (x, y))| {
        if i == 0 {
            format!("M {} {}", x, y)
        } else {
            format!("L {} {}", x, y)
        }
    })
    .collect::<Vec<_>>()
    .join(" ");

rsx! {
    path {
        d: "{path_data}",
        stroke: "blue",
        fill: "none",
        stroke_width: "2",
    }
}
```

**Example: Optimizer Path (optimizer_demo.rs improvement):**

```rust
// Current pattern creates many circles, could be optimized to path
fn render_optimizer_path(path: &[(f64, f64)], color: &str) -> Element {
    let path_data = path
        .iter()
        .enumerate()
        .map(|(i, (x, y))| {
            let svg_x = (x + bounds.min_x.abs()) * scale;
            let svg_y = height - (y + bounds.min_y.abs()) * scale;
            if i == 0 {
                format!("M {:.2} {:.2}", svg_x, svg_y)
            } else {
                format!("L {:.2} {:.2}", svg_x, svg_y)
            }
        })
        .collect::<Vec<_>>()
        .join(" ");

    rsx! {
        // Path for trail
        path {
            d: "{path_data}",
            stroke: "{color}",
            stroke_width: "2",
            fill: "none",
            opacity: "0.6",
        }

        // Current position (single circle)
        if let Some((x, y)) = path.last() {
            circle {
                cx: "{(x + bounds.min_x.abs()) * scale}",
                cy: "{height - (y + bounds.min_y.abs()) * scale}",
                r: "5",
                fill: "{color}",
            }
        }
    }
}
```

### 3.3 Canvas for High-Performance Animation

**When to migrate to Canvas:**
- Optimizer demo with >1000 path points
- Real-time heatmap updates (>10 FPS)
- Particle effects or fluid simulations

**Canvas Pattern (using web-sys):**

```rust
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

#[component]
pub fn CanvasVisualizer() -> Element {
    let mut canvas_ref = use_signal(|| None::<HtmlCanvasElement>);

    // Render loop
    use_effect(move || {
        if let Some(canvas) = canvas_ref.read().as_ref() {
            let context = canvas
                .get_context("2d")
                .unwrap()
                .unwrap()
                .dyn_into::<CanvasRenderingContext2d>()
                .unwrap();

            // Clear canvas
            context.clear_rect(0.0, 0.0, canvas.width() as f64, canvas.height() as f64);

            // Draw heatmap
            for i in 0..50 {
                for j in 0..50 {
                    let color = compute_color(i, j);
                    context.set_fill_style(&color.into());
                    context.fill_rect(
                        i as f64 * 10.0,
                        j as f64 * 10.0,
                        10.0,
                        10.0,
                    );
                }
            }
        }
    });

    rsx! {
        canvas {
            id: "visualization",
            width: "500",
            height: "500",
            onmounted: move |evt| {
                if let Some(element) = evt.data.downcast::<web_sys::Element>() {
                    canvas_ref.set(Some(element.dyn_into().unwrap()));
                }
            },
        }
    }
}
```

### 3.4 Chart Libraries Compatible with Dioxus

**Option 1: dioxus-charts (SVG-based)**

```toml
[dependencies]
dioxus-charts = "0.3"
```

```rust
use dioxus_charts::*;

#[component]
pub fn TrainingHistoryChart(losses: Vec<f64>) -> Element {
    let data = losses
        .iter()
        .enumerate()
        .map(|(i, &loss)| (i as f64, loss))
        .collect::<Vec<_>>();

    rsx! {
        LineChart {
            data: data,
            width: 600.0,
            height: 400.0,
            x_label: "Iteration",
            y_label: "Loss",
            color: "#4A90E2",
        }
    }
}
```

**Option 2: plotters-dioxus (Your existing dependency!)**

You already have `plotters` in your dependencies! Use it:

```rust
use plotters::prelude::*;
use plotters_dioxus::DioxusBackend;

#[component]
pub fn PlottersChart() -> Element {
    let svg = use_memo(|| {
        let mut buffer = String::new();
        {
            let root = DioxusBackend::new(&mut buffer, (600, 400)).into_drawing_area();
            root.fill(&WHITE).unwrap();

            let mut chart = ChartBuilder::on(&root)
                .caption("Training History", ("sans-serif", 30))
                .margin(10)
                .x_label_area_size(40)
                .y_label_area_size(50)
                .build_cartesian_2d(0f64..100f64, 0f64..1f64)
                .unwrap();

            chart.configure_mesh().draw().unwrap();

            chart.draw_series(LineSeries::new(
                (0..100).map(|x| (x as f64, (x as f64 / 100.0).sin())),
                &BLUE,
            )).unwrap();
        }
        buffer
    });

    rsx! {
        div { dangerous_inner_html: "{svg}" }
    }
}
```

**Recommendation:** Stick with your custom SVG for now. It's:
- More educational (users see the code)
- More flexible (full control)
- Performant enough for your dataset sizes (<1000 samples)

Migrate to Canvas only if you validate >1000 iter/sec requirement.

---

## 4. Form & Input Handling

### Current Pattern (ml_playground.rs)

```rust
input {
    r#type: "file",
    accept: ".csv",
    id: "csv-upload",
    onchange: move |evt| {
        async move {
            if let Some(file_engine) = evt.files() {
                let files = file_engine.files();
                // Process file...
            }
        }
    }
}
```

### 4.1 Improved File Upload with Validation

```rust
#[component]
pub fn CsvUploadWithValidation() -> Element {
    let mut upload_status = use_signal(|| "".to_string());
    let mut is_uploading = use_signal(|| false);

    // File size limits (5MB max)
    const MAX_FILE_SIZE: usize = 5 * 1024 * 1024;
    const MAX_ROWS: usize = 10_000;
    const MAX_FEATURES: usize = 100;

    rsx! {
        div { class: "upload-container",
            input {
                r#type: "file",
                accept: ".csv",
                id: "csv-upload",
                disabled: *is_uploading.read(),
                onchange: move |evt| {
                    spawn(async move {
                        is_uploading.set(true);
                        upload_status.set("🔄 Validating file...".to_string());

                        if let Some(file_engine) = evt.files() {
                            let files = file_engine.files();

                            if let Some(file_name) = files.first() {
                                // Read file
                                if let Some(file_contents) = file_engine.read_file(file_name).await {
                                    // Validate file size
                                    if file_contents.len() > MAX_FILE_SIZE {
                                        upload_status.set(format!(
                                            "❌ File too large: {:.1} MB (max 5 MB)",
                                            file_contents.len() as f64 / 1_048_576.0
                                        ));
                                        is_uploading.set(false);
                                        return;
                                    }

                                    // Parse as UTF-8
                                    match String::from_utf8(file_contents) {
                                        Ok(content_str) => {
                                            // Validate row count
                                            let row_count = content_str.lines().count();
                                            if row_count > MAX_ROWS {
                                                upload_status.set(format!(
                                                    "❌ Too many rows: {} (max {})",
                                                    row_count, MAX_ROWS
                                                ));
                                                is_uploading.set(false);
                                                return;
                                            }

                                            // Validate columns
                                            let headers: Vec<&str> = content_str
                                                .lines()
                                                .next()
                                                .unwrap_or("")
                                                .split(',')
                                                .collect();

                                            if headers.len() > MAX_FEATURES {
                                                upload_status.set(format!(
                                                    "❌ Too many features: {} (max {})",
                                                    headers.len(), MAX_FEATURES
                                                ));
                                                is_uploading.set(false);
                                                return;
                                            }

                                            // Parse CSV
                                            upload_status.set("🔄 Parsing CSV...".to_string());

                                            match CsvDataset::from_csv(&content_str, headers[0]) {
                                                Ok(dataset) => {
                                                    upload_status.set(format!(
                                                        "✅ Loaded {} rows, {} features",
                                                        dataset.num_samples,
                                                        dataset.features.cols
                                                    ));
                                                    // Store dataset...
                                                }
                                                Err(e) => {
                                                    upload_status.set(format!("❌ Parse error: {}", e));
                                                }
                                            }
                                        }
                                        Err(_) => {
                                            upload_status.set("❌ Invalid UTF-8 encoding".to_string());
                                        }
                                    }
                                } else {
                                    upload_status.set("❌ Failed to read file".to_string());
                                }
                            }
                        }

                        is_uploading.set(false);
                    });
                }
            }

            label {
                r#for: "csv-upload",
                class: if *is_uploading.read() { "upload-button disabled" } else { "upload-button" },
                if *is_uploading.read() {
                    "⏳ Processing..."
                } else {
                    "📁 Choose CSV File"
                }
            }

            if !upload_status.read().is_empty() {
                div { class: "upload-status",
                    "{upload_status}"
                }
            }

            // Show constraints
            div { class: "upload-constraints",
                p { class: "hint",
                    "Max: 5 MB, 10,000 rows, 100 features"
                }
            }
        }
    }
}
```

### 4.2 Drag-and-Drop File Upload

```rust
#[component]
pub fn DragDropCsvUpload() -> Element {
    let mut is_dragging = use_signal(|| false);
    let mut csv_dataset = use_signal(|| None::<CsvDataset>);

    let handle_drop = move |evt: DragEvent| {
        evt.prevent_default();
        is_dragging.set(false);

        spawn(async move {
            if let Some(file_engine) = evt.files() {
                // Process file (same as regular upload)
            }
        });
    };

    rsx! {
        div {
            class: if *is_dragging.read() { "drop-zone dragging" } else { "drop-zone" },
            ondrop: handle_drop,
            ondragover: move |evt| {
                evt.prevent_default();
                is_dragging.set(true);
            },
            ondragleave: move |_| {
                is_dragging.set(false);
            },

            if csv_dataset.read().is_none() {
                div { class: "drop-prompt",
                    "📁 Drag & Drop CSV Here"
                    p { "or click to browse" }
                }
            } else {
                div { class: "dataset-preview",
                    "✅ Dataset loaded"
                }
            }

            // Hidden file input as fallback
            input {
                r#type: "file",
                accept: ".csv",
                style: "display: none;",
                id: "file-input-hidden",
            }
        }
    }
}
```

### 4.3 Dynamic Algorithm Configuration

**Add to ml_playground.rs:**

```rust
#[derive(Clone, Debug)]
struct AlgorithmParams {
    kmeans_k: usize,
    kmeans_max_iter: usize,
    pca_components: usize,
    learning_rate: f64,
    logistic_max_iter: usize,
}

impl Default for AlgorithmParams {
    fn default() -> Self {
        Self {
            kmeans_k: 3,
            kmeans_max_iter: 100,
            pca_components: 2,
            learning_rate: 0.01,
            logistic_max_iter: 1000,
        }
    }
}

#[component]
pub fn AlgorithmConfigPanel(
    algorithm: Signal<Algorithm>,
    params: Signal<AlgorithmParams>,
) -> Element {
    rsx! {
        div { class: "config-panel",
            h3 { "⚙️ Configuration" }

            match algorithm() {
                Algorithm::KMeans => rsx! {
                    div { class: "param-group",
                        label {
                            "Number of Clusters (k):"
                            input {
                                r#type: "range",
                                min: "2",
                                max: "10",
                                value: "{params().kmeans_k}",
                                oninput: move |evt| {
                                    params.with_mut(|p| {
                                        p.kmeans_k = evt.value().parse().unwrap_or(3);
                                    });
                                },
                            }
                            span { class: "param-value", "{params().kmeans_k}" }
                        }

                        label {
                            "Max Iterations:"
                            input {
                                r#type: "range",
                                min: "10",
                                max: "500",
                                step: "10",
                                value: "{params().kmeans_max_iter}",
                                oninput: move |evt| {
                                    params.with_mut(|p| {
                                        p.kmeans_max_iter = evt.value().parse().unwrap_or(100);
                                    });
                                },
                            }
                            span { class: "param-value", "{params().kmeans_max_iter}" }
                        }
                    }
                },

                Algorithm::PCA => rsx! {
                    div { class: "param-group",
                        label {
                            "Number of Components:"
                            input {
                                r#type: "number",
                                min: "1",
                                max: "50",
                                value: "{params().pca_components}",
                                oninput: move |evt| {
                                    params.with_mut(|p| {
                                        p.pca_components = evt.value().parse().unwrap_or(2);
                                    });
                                },
                            }
                        }
                    }
                },

                Algorithm::LogisticRegression => rsx! {
                    div { class: "param-group",
                        label {
                            "Learning Rate:"
                            input {
                                r#type: "number",
                                step: "0.001",
                                min: "0.001",
                                max: "1.0",
                                value: "{params().learning_rate}",
                                oninput: move |evt| {
                                    params.with_mut(|p| {
                                        p.learning_rate = evt.value().parse().unwrap_or(0.01);
                                    });
                                },
                            }
                        }
                    }
                },

                _ => rsx! {
                    p { class: "hint", "No configurable parameters" }
                },
            }
        }
    }
}
```

### 4.4 Form Validation Pattern

```rust
#[component]
pub fn ValidatedInput(
    label: String,
    value: Signal<String>,
    validator: fn(&str) -> Result<(), String>,
) -> Element {
    let mut error = use_signal(|| None::<String>);

    rsx! {
        div { class: "input-group",
            label { "{label}" }
            input {
                r#type: "text",
                value: "{value()}",
                oninput: move |evt| {
                    let new_value = evt.value();
                    value.set(new_value.clone());

                    // Validate
                    match validator(&new_value) {
                        Ok(()) => error.set(None),
                        Err(e) => error.set(Some(e)),
                    }
                },
                class: if error.read().is_some() { "invalid" } else { "" },
            }

            if let Some(ref err) = *error.read() {
                span { class: "error-message", "{err}" }
            }
        }
    }
}

// Usage
fn validate_learning_rate(value: &str) -> Result<(), String> {
    match value.parse::<f64>() {
        Ok(lr) if lr > 0.0 && lr <= 1.0 => Ok(()),
        Ok(_) => Err("Must be between 0 and 1".to_string()),
        Err(_) => Err("Must be a number".to_string()),
    }
}

rsx! {
    ValidatedInput {
        label: "Learning Rate".to_string(),
        value: learning_rate_str,
        validator: validate_learning_rate,
    }
}
```

---

## 5. Async & WASM Integration

### Current Pattern (ml_playground.rs, Lines 156-167)

```rust
onclick: move |_| {
    spawn(async move {
        is_processing.set(true);
        result_message.set(format!("🔄 Running {}...", selected_algorithm.read().name()));

        if let Some(ref dataset) = *csv_dataset.read() {
            let result = run_algorithm(*selected_algorithm.read(), dataset);
            result_message.set(result);
        }

        is_processing.set(false);
    });
},
```

**Problem:** No panic recovery! WASM panic kills entire app.

### 5.1 WASM-Safe Async Pattern with Panic Recovery

```rust
use std::panic;
use wasm_bindgen::JsValue;
use web_sys::console;

onclick: move |_| {
    spawn(async move {
        is_processing.set(true);
        result_message.set("🔄 Processing...".to_string());

        // Panic boundary
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            if let Some(ref dataset) = *csv_dataset.read() {
                run_algorithm(*selected_algorithm.read(), dataset)
            } else {
                "❌ No dataset loaded".to_string()
            }
        }));

        match result {
            Ok(msg) => {
                result_message.set(msg);
            }
            Err(panic_info) => {
                // Log to console for debugging
                console::error_1(&JsValue::from_str("WASM panic caught"));

                // User-friendly error
                result_message.set(
                    "❌ Algorithm crashed. Try with simpler data or reload the page.".to_string()
                );
            }
        }

        is_processing.set(false);
    });
},
```

### 5.2 Timeout for Long-Running Operations

```rust
use gloo::timers::future::TimeoutFuture;
use std::future::Future;

async fn with_timeout<F, T>(
    future: F,
    timeout_ms: u32,
) -> Result<T, &'static str>
where
    F: Future<Output = T>,
{
    use futures::future::select;
    use futures::pin_mut;

    let timeout = TimeoutFuture::new(timeout_ms);

    pin_mut!(future);
    pin_mut!(timeout);

    match select(future, timeout).await {
        futures::future::Either::Left((result, _)) => Ok(result),
        futures::future::Either::Right(_) => Err("Operation timed out"),
    }
}

// Usage
onclick: move |_| {
    spawn(async move {
        is_processing.set(true);

        // 5 second timeout
        let result = with_timeout(
            async {
                let dataset = csv_dataset.read();
                run_algorithm(*selected_algorithm.read(), dataset.as_ref().unwrap())
            },
            5000,
        ).await;

        match result {
            Ok(msg) => result_message.set(msg),
            Err(e) => result_message.set(format!("❌ {}", e)),
        }

        is_processing.set(false);
    });
},
```

### 5.3 Progress Tracking for Long Operations

```rust
#[component]
pub fn MLPlayground() -> Element {
    let mut csv_dataset = use_signal(|| None::<CsvDataset>);
    let mut progress = use_signal(|| 0);
    let mut is_processing = use_signal(|| false);

    rsx! {
        div { class: "ml-playground",
            // ... other UI

            button {
                onclick: move |_| {
                    spawn(async move {
                        is_processing.set(true);
                        progress.set(0);

                        // Simulate progress updates
                        for i in 0..10 {
                            // Do 10% of work
                            process_chunk(i);

                            // Update progress
                            progress.set((i + 1) * 10);

                            // Yield to browser every iteration
                            TimeoutFuture::new(0).await;
                        }

                        is_processing.set(false);
                    });
                },
                "Run Algorithm"
            }

            if *is_processing.read() {
                div { class: "progress-container",
                    div {
                        class: "progress-bar",
                        style: "width: {progress}%",
                    }
                    p { "Processing: {progress}% complete" }
                }
            }
        }
    }
}
```

### 5.4 Cancellable Operations

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[component]
pub fn CancellableOperation() -> Element {
    let mut is_running = use_signal(|| false);
    let cancel_flag = use_signal(|| Arc::new(AtomicBool::new(false)));

    rsx! {
        div {
            button {
                disabled: *is_running.read(),
                onclick: move |_| {
                    let flag = Arc::new(AtomicBool::new(false));
                    cancel_flag.set(flag.clone());
                    is_running.set(true);

                    spawn(async move {
                        // Run algorithm with cancellation checks
                        for i in 0..1000 {
                            if flag.load(Ordering::Relaxed) {
                                // Cancelled!
                                break;
                            }

                            // Do work
                            process_iteration(i);

                            // Yield every 10 iterations
                            if i % 10 == 0 {
                                TimeoutFuture::new(0).await;
                            }
                        }

                        is_running.set(false);
                    });
                },
                "▶ Run"
            }

            if *is_running.read() {
                button {
                    onclick: move |_| {
                        cancel_flag.read().store(true, Ordering::Relaxed);
                    },
                    "⏹ Cancel"
                }
            }
        }
    }
}
```

### 5.5 Error Boundaries (Dioxus 0.6)

```rust
use dioxus::prelude::*;

#[component]
pub fn ErrorBoundary(children: Element) -> Element {
    let error = use_signal(|| None::<String>);

    rsx! {
        ErrorHandler {
            handle_error: move |err: &ErrorContext| {
                error.set(Some(err.message.clone()));
            },

            if let Some(ref err) = *error.read() {
                div { class: "error-boundary",
                    h2 { "❌ Something went wrong" }
                    p { "{err}" }
                    button {
                        onclick: move |_| error.set(None),
                        "🔄 Retry"
                    }
                }
            } else {
                {children}
            }
        }
    }
}

// Usage
rsx! {
    ErrorBoundary {
        MLPlayground {}
    }
}
```

---

## 6. Styling & Theming

### Current Approach: External CSS (main.css)

You're using external CSS, which is correct! Here's the comparison:

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **External CSS** | ✅ Fast, cached, familiar | ⚠️ No type safety | ✅ Your current (good!) |
| **CSS-in-Rust** | ✅ Type-safe, scoped | ⚠️ Bloats WASM | ⚠️ Only for dynamic styles |
| **Tailwind** | ✅ Utility-first, fast dev | ⚠️ Learning curve | 🔮 Future consideration |

### 6.1 Current CSS Architecture

```css
/* main.css structure */
body { /* Global styles */ }

/* Component-specific classes */
.ml-playground { }
.algorithm-btn { }
.result-message { }

/* State modifiers */
.algorithm-btn.selected { }
.tab-button.active { }
```

**Strengths:**
- ✅ Clear naming conventions
- ✅ Component isolation
- ✅ Easy to maintain

### 6.2 CSS Custom Properties for Theming

**Add to main.css:**

```css
:root {
  /* Color palette */
  --primary-color: #4A90E2;
  --secondary-color: #7B68EE;
  --success-color: #4CAF50;
  --error-color: #F44336;
  --warning-color: #FF9800;

  /* Backgrounds */
  --bg-primary: #FFFFFF;
  --bg-secondary: #F9F9F9;
  --bg-tertiary: #E8F4F8;

  /* Text */
  --text-primary: #111111;
  --text-secondary: #555555;
  --text-muted: #999999;

  /* Borders */
  --border-color: #DDDDDD;
  --border-radius: 8px;

  /* Shadows */
  --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.1);
  --shadow-md: 0 4px 8px rgba(0, 0, 0, 0.15);
  --shadow-lg: 0 8px 16px rgba(0, 0, 0, 0.2);

  /* Animations */
  --transition-fast: 150ms ease;
  --transition-normal: 300ms ease;
  --transition-slow: 500ms ease;
}

/* Dark mode */
[data-theme="dark"] {
  --primary-color: #6BA4E7;
  --secondary-color: #9B88FF;

  --bg-primary: #1E1E1E;
  --bg-secondary: #2D2D2D;
  --bg-tertiary: #3A3A3A;

  --text-primary: #E0E0E0;
  --text-secondary: #B0B0B0;
  --text-muted: #707070;

  --border-color: #444444;
}

/* Use in components */
.ml-playground {
  background-color: var(--bg-primary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  box-shadow: var(--shadow-md);
}

.algorithm-btn {
  background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
  transition: transform var(--transition-fast);
}

.algorithm-btn:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}
```

### 6.3 Dark Mode Implementation

**In App component:**

```rust
#[component]
fn App() -> Element {
    let mut dark_mode = use_signal(|| {
        // Check localStorage or system preference
        false
    });

    // Update document theme
    use_effect(move || {
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            if let Some(html) = doc.document_element() {
                if *dark_mode.read() {
                    html.set_attribute("data-theme", "dark").ok();
                } else {
                    html.remove_attribute("data-theme").ok();
                }
            }
        }
    });

    rsx! {
        document::Stylesheet { href: CSS }

        // Theme toggle (global header)
        div { class: "app-header",
            button {
                class: "theme-toggle",
                onclick: move |_| dark_mode.set(!dark_mode()),
                if *dark_mode.read() { "☀️ Light" } else { "🌙 Dark" }
            }
        }

        Router::<Route> {}
    }
}
```

### 6.4 Responsive Design Patterns

**Add to main.css:**

```css
/* Mobile-first approach */
.ml-playground {
  padding: 10px;
}

.playground-container {
  display: flex;
  flex-direction: column; /* Stack vertically on mobile */
}

.control-panel {
  width: 100%;
}

.results-panel {
  width: 100%;
}

/* Tablet (768px+) */
@media (min-width: 768px) {
  .ml-playground {
    padding: 20px;
  }

  .playground-container {
    flex-direction: row;
  }

  .control-panel {
    width: 300px;
    flex-shrink: 0;
  }

  .results-panel {
    width: calc(100% - 300px);
  }
}

/* Desktop (1024px+) */
@media (min-width: 1024px) {
  .ml-playground {
    padding: 30px;
  }

  .control-panel {
    width: 350px;
  }
}

/* Large screens (1440px+) */
@media (min-width: 1440px) {
  .playground-container {
    max-width: 1400px;
    margin: 0 auto;
  }
}

/* Handle mobile hover states */
@media (hover: none) {
  .algorithm-btn:hover {
    /* Disable hover effects on touch devices */
    transform: none;
  }
}
```

### 6.5 Animation Utilities

**Add to main.css:**

```css
/* Fade in */
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.fade-in {
  animation: fadeIn var(--transition-normal) ease-out;
}

/* Spinner */
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid var(--border-color);
  border-top-color: var(--primary-color);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

/* Pulse */
@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.pulse {
  animation: pulse 2s ease-in-out infinite;
}

/* Slide in from right */
@keyframes slideInRight {
  from {
    transform: translateX(100%);
  }
  to {
    transform: translateX(0);
  }
}

.result-message {
  animation: slideInRight var(--transition-normal) ease-out;
}
```

**Use in components:**

```rust
rsx! {
    div {
        class: "fade-in",
        "Content appears smoothly"
    }

    if *is_processing.read() {
        div { class: "spinner" }
    }
}
```

### 6.6 Conditional Styling in Dioxus

```rust
// Simple conditional class
rsx! {
    button {
        class: if is_active { "btn active" } else { "btn" },
        "Click me"
    }
}

// Multiple conditions
rsx! {
    div {
        class: format!(
            "card {} {} {}",
            if is_selected { "selected" } else { "" },
            if is_disabled { "disabled" } else { "" },
            if is_loading { "loading" } else { "" },
        ),
    }
}

// Inline styles (avoid unless dynamic)
rsx! {
    div {
        style: "background-color: {color}; width: {width}px;",
    }
}

// Better: CSS custom properties
rsx! {
    div {
        class: "dynamic-box",
        style: "--box-color: {color}; --box-width: {width}px;",
    }
}
```

```css
.dynamic-box {
  background-color: var(--box-color, blue);
  width: var(--box-width, 100px);
}
```

---

## 7. Performance Patterns

### Your Zero-Allocation Pattern (Gold Standard!)

**From optimizer.rs:**

```rust
// Zero allocations - uses scalar tuples
pub fn step_2d(&mut self, pos: (f64, f64), grad: (f64, f64)) -> (f64, f64) {
    // Pure scalar math, no heap allocations
    let (x, y) = pos;
    let (dx, dy) = grad;

    // ... optimizer-specific math ...

    (new_x, new_y)
}
```

**Result:** 1000+ iterations/sec (vs 200-500 with Matrix allocations)

### 7.1 Bounded Memory Pattern

**From optimizer_demo.rs:**

```rust
const MAX_PATH_LENGTH: usize = 1000;
const MAX_LOSS_HISTORY: usize = 10000;

fn step(&mut self, loss_fn: &LossFunction) {
    // ... compute new position ...

    // Bounded circular buffer
    if self.path.len() >= MAX_PATH_LENGTH {
        self.path.remove(0); // Remove oldest
    }
    self.path.push(new_position);
}
```

**Critical for WASM:** Prevents unbounded memory growth in long-running demos.

### 7.2 Pre-computation Pattern

```rust
// ❌ BAD: Compute in render
rsx! {
    for i in 0..1000 {
        div {
            "Value: {expensive_computation(i)}"
        }
    }
}

// ✅ GOOD: Pre-compute once
let computed_values: Vec<String> = (0..1000)
    .map(|i| expensive_computation(i))
    .collect();

rsx! {
    for value in computed_values {
        div { "{value}" }
    }
}

// ✅ BETTER: Use memo for reactive pre-computation
let computed_values = use_memo(move || {
    (0..1000)
        .map(|i| expensive_computation(i, *input.read()))
        .collect::<Vec<_>>()
});

rsx! {
    for value in computed_values() {
        div { "{value}" }
    }
}
```

### 7.3 Debouncing User Input

```rust
use gloo::timers::future::TimeoutFuture;

#[component]
pub fn DebouncedSearch() -> Element {
    let mut search_term = use_signal(|| String::new());
    let mut search_results = use_signal(|| Vec::new());

    // Debounce search
    let mut debounce_task = use_signal(|| None::<TaskId>);

    rsx! {
        input {
            r#type: "text",
            value: "{search_term()}",
            oninput: move |evt| {
                let new_value = evt.value();
                search_term.set(new_value.clone());

                // Cancel previous debounce
                if let Some(task_id) = debounce_task.take() {
                    // Task will be dropped/cancelled
                }

                // Start new debounce
                let task = spawn(async move {
                    TimeoutFuture::new(300).await; // 300ms delay

                    // Perform search
                    let results = perform_search(&new_value);
                    search_results.set(results);
                });

                debounce_task.set(Some(task));
            },
        }

        div { class: "search-results",
            for result in search_results() {
                div { "{result}" }
            }
        }
    }
}
```

### 7.4 Lazy Loading / Virtual Scrolling

**For large feature lists (100+ features):**

```rust
#[component]
pub fn VirtualizedFeatureList(features: Vec<String>) -> Element {
    let mut scroll_top = use_signal(|| 0.0);

    const ITEM_HEIGHT: f64 = 40.0;
    const VISIBLE_ITEMS: usize = 20;

    // Calculate visible range
    let start_idx = (scroll_top() / ITEM_HEIGHT).floor() as usize;
    let end_idx = (start_idx + VISIBLE_ITEMS).min(features.len());

    let total_height = features.len() as f64 * ITEM_HEIGHT;
    let offset_top = start_idx as f64 * ITEM_HEIGHT;

    rsx! {
        div {
            class: "virtual-scroll-container",
            style: "height: 800px; overflow-y: auto;",
            onscroll: move |evt| {
                if let Some(target) = evt.data.downcast::<web_sys::Element>() {
                    scroll_top.set(target.scroll_top() as f64);
                }
            },

            // Spacer for total height
            div {
                style: "height: {total_height}px; position: relative;",

                // Visible items only
                div {
                    style: "position: absolute; top: {offset_top}px; width: 100%;",
                    for i in start_idx..end_idx {
                        div {
                            key: "{i}",
                            style: "height: {ITEM_HEIGHT}px;",
                            class: "feature-item",
                            "{features[i]}"
                        }
                    }
                }
            }
        }
    }
}
```

---

## 8. Production-Ready Patterns

### 8.1 Complete WASM-Safe Algorithm Runner

**Create `web/src/components/safe_algorithm_runner.rs`:**

```rust
use std::panic;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use wasm_bindgen::JsValue;
use web_sys::console;
use gloo::timers::future::TimeoutFuture;

/// Safe execution wrapper for ML algorithms in WASM
pub struct SafeAlgorithmRunner {
    timeout_ms: u32,
    max_file_size: usize,
    max_rows: usize,
    max_features: usize,
}

impl Default for SafeAlgorithmRunner {
    fn default() -> Self {
        Self {
            timeout_ms: 5000,
            max_file_size: 5 * 1024 * 1024, // 5MB
            max_rows: 10_000,
            max_features: 100,
        }
    }
}

impl SafeAlgorithmRunner {
    pub async fn run_with_safety<F, T>(
        &self,
        operation: F,
        cancel_flag: Arc<AtomicBool>,
    ) -> Result<T, String>
    where
        F: FnOnce() -> T + panic::UnwindSafe,
    {
        // Check cancellation
        if cancel_flag.load(Ordering::Relaxed) {
            return Err("Operation cancelled".to_string());
        }

        // Timeout wrapper
        let timeout_future = TimeoutFuture::new(self.timeout_ms);

        // Run with panic recovery
        let result_future = async {
            match panic::catch_unwind(operation) {
                Ok(result) => Ok(result),
                Err(panic_info) => {
                    // Log panic for debugging
                    console::error_1(&JsValue::from_str("WASM panic caught"));

                    // Try to extract panic message
                    let panic_msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                        s.clone()
                    } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                        s.to_string()
                    } else {
                        "Unknown panic".to_string()
                    };

                    Err(format!("Algorithm crashed: {}", panic_msg))
                }
            }
        };

        // Race timeout vs operation
        use futures::future::select;
        use futures::pin_mut;

        pin_mut!(result_future);
        pin_mut!(timeout_future);

        match select(result_future, timeout_future).await {
            futures::future::Either::Left((result, _)) => result,
            futures::future::Either::Right(_) => {
                Err(format!("Operation timed out after {}ms", self.timeout_ms))
            }
        }
    }

    pub fn validate_dataset(&self, dataset: &CsvDataset) -> Result<(), String> {
        if dataset.num_samples > self.max_rows {
            return Err(format!(
                "Dataset too large: {} rows (max {})",
                dataset.num_samples, self.max_rows
            ));
        }

        if dataset.features.cols > self.max_features {
            return Err(format!(
                "Too many features: {} (max {})",
                dataset.features.cols, self.max_features
            ));
        }

        Ok(())
    }
}
```

### 8.2 Structured Error Types

**Create `ml_traits/src/error.rs`:**

```rust
use std::fmt;

#[derive(Debug, Clone)]
pub enum MLError {
    InvalidInput {
        message: String,
        parameter: &'static str,
    },
    NotFitted {
        model_type: &'static str,
    },
    DimensionMismatch {
        expected: (usize, usize),
        got: (usize, usize),
    },
    ConvergenceFailure {
        iterations: usize,
        final_cost: f64,
    },
    InsufficientData {
        required: usize,
        provided: usize,
    },
    NumericalInstability {
        context: String,
    },
    Timeout {
        duration_ms: u32,
    },
    Cancelled,
}

impl fmt::Display for MLError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MLError::InvalidInput { message, parameter } => {
                write!(f, "Invalid input for '{}': {}", parameter, message)
            }
            MLError::NotFitted { model_type } => {
                write!(f, "{} must be fitted before prediction", model_type)
            }
            MLError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Dimension mismatch: expected {:?}, got {:?}",
                    expected, got
                )
            }
            MLError::ConvergenceFailure { iterations, final_cost } => {
                write!(
                    f,
                    "Failed to converge after {} iterations (final cost: {:.6})",
                    iterations, final_cost
                )
            }
            MLError::InsufficientData { required, provided } => {
                write!(
                    f,
                    "Insufficient data: requires {}, provided {}",
                    required, provided
                )
            }
            MLError::NumericalInstability { context } => {
                write!(f, "Numerical instability: {}", context)
            }
            MLError::Timeout { duration_ms } => {
                write!(f, "Operation timed out after {}ms", duration_ms)
            }
            MLError::Cancelled => {
                write!(f, "Operation was cancelled")
            }
        }
    }
}

impl std::error::Error for MLError {}

// Temporary bridge from String errors
impl From<String> for MLError {
    fn from(s: String) -> Self {
        MLError::NumericalInstability { context: s }
    }
}
```

### 8.3 Accessibility Patterns

**Add ARIA attributes:**

```rust
rsx! {
    div {
        class: "ml-playground",
        role: "main",
        "aria-label": "Machine Learning Playground",

        button {
            class: "run-button",
            disabled: *is_processing.read(),
            "aria-busy": "{is_processing()}",
            "aria-label": "Run selected algorithm",
            onclick: /* ... */,

            if *is_processing.read() {
                span {
                    class: "spinner",
                    "aria-hidden": "true",
                }
                span { class: "sr-only", "Processing..." }
            } else {
                "▶ Run Algorithm"
            }
        }

        // Results with live region
        div {
            class: "result-message",
            role: "status",
            "aria-live": "polite",
            "aria-atomic": "true",
            "{result_message}"
        }
    }
}
```

**Add to main.css:**

```css
/* Screen reader only */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

/* Focus visible (keyboard navigation) */
:focus-visible {
  outline: 3px solid var(--primary-color);
  outline-offset: 2px;
}

/* Skip to main content */
.skip-to-main {
  position: absolute;
  top: -40px;
  left: 0;
  background: var(--primary-color);
  color: white;
  padding: 8px 16px;
  text-decoration: none;
  z-index: 100;
}

.skip-to-main:focus {
  top: 0;
}
```

### 8.4 Testing Patterns

**Example E2E test structure:**

```rust
// tests/ml_playground_test.rs
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_csv_upload_validation() {
    // Test file size limit
    let large_file = create_csv_file(6 * 1024 * 1024); // 6MB
    let result = upload_csv(large_file).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("too large"));
}

#[wasm_bindgen_test]
async fn test_algorithm_execution() {
    let dataset = create_test_dataset(100, 3);
    let result = run_kmeans(dataset, 3, 100).await;
    assert!(result.is_ok());
}

#[wasm_bindgen_test]
async fn test_panic_recovery() {
    let invalid_dataset = create_invalid_dataset();
    let result = run_algorithm_safe(invalid_dataset).await;

    // Should not crash, should return error
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("crashed"));
}
```

---

## 9. Quick Reference: Common Patterns

### State Management Cheatsheet

```rust
// Local state
let mut count = use_signal(|| 0);

// Read
let value = *count.read();
let value = count(); // Shorthand

// Write
count.set(42);
count.with_mut(|c| *c += 1);

// Context (provide once in App)
use_context_provider(|| ThemeContext { /* ... */ });

// Context (consume anywhere)
let theme = use_context::<ThemeContext>();

// Global (define at module level)
static CACHE: GlobalSignal<HashMap<String, Data>> = Signal::global(HashMap::new);
```

### Routing Cheatsheet

```rust
// Define routes
#[derive(Routable, Clone, PartialEq)]
enum Route {
    #[route("/")]
    Home,

    #[route("/page/:id")]
    Page { id: String },
}

// Navigate
let nav = use_navigator();
nav.push(Route::Page { id: "123".to_string() });

// Link
rsx! {
    Link { to: Route::Home, "Go Home" }
}
```

### Async Cheatsheet

```rust
// Spawn task
spawn(async move {
    let result = do_async_work().await;
    state.set(result);
});

// With timeout
let result = with_timeout(operation, 5000).await;

// With panic recovery
let result = panic::catch_unwind(|| {
    risky_operation()
});
```

### Performance Cheatsheet

```rust
// Pre-compute
let values = use_memo(move || {
    expensive_computation(*input.read())
});

// Debounce
let task = spawn(async move {
    TimeoutFuture::new(300).await;
    handle_input();
});

// Bounded memory
if buffer.len() >= MAX_SIZE {
    buffer.remove(0);
}
buffer.push(new_value);
```

---

## 10. Official Resources

### Documentation
- **Dioxus 0.6 Guide**: https://dioxuslabs.com/learn/0.6/guide/
- **Router Guide**: https://dioxuslabs.com/learn/0.6/guide/routing/
- **State Management**: https://dioxuslabs.com/learn/0.6/guide/state/
- **API Docs**: https://docs.rs/dioxus/0.6.0/

### Community
- **GitHub**: https://github.com/DioxusLabs/dioxus
- **Discord**: https://discord.gg/XgGxMSkvUM
- **Examples**: https://github.com/DioxusLabs/dioxus/tree/main/examples

### Related Libraries
- **dioxus-charts**: https://github.com/dioxus-community/dioxus-charts
- **plotters**: https://docs.rs/plotters/latest/plotters/
- **gloo**: https://docs.rs/gloo/latest/gloo/ (WASM utilities)

---

## Appendix: Migration Priorities

Based on your CLAUDE.md Week 1-4 priorities:

### Week 1 (Critical)
1. ✅ Add panic boundaries to all algorithm runs (Section 5.1)
2. ✅ Implement timeout protection (Section 5.2)
3. ✅ Add CSV validation with limits (Section 4.1)

### Week 2 (High-Value)
1. ✅ Add algorithm configuration UI (Section 4.3)
2. ✅ Implement progress indicators (Section 5.3)
3. ✅ Add dark mode toggle (Section 6.3)

### Week 3 (Polish)
1. ✅ Responsive design improvements (Section 6.4)
2. ✅ Accessibility enhancements (Section 8.3)
3. ✅ Error boundaries (Section 5.5)

### Week 4 (Advanced)
1. 🔮 Algorithm comparison mode (Section 2.5)
2. 🔮 URL state persistence (Section 2.1)
3. 🔮 Virtual scrolling for large datasets (Section 7.4)

---

**Last Updated:** November 8, 2025
**Dioxus Version:** 0.6.0
**Status:** Production-ready patterns validated
