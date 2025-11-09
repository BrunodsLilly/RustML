# UI Overhaul Quick Reference
**Fast lookup guide for UI architecture analysis**

## Current Component Inventory

### Core Pages (5 routes)
- `/` - MainView (landing)
- `/showcase` - Matrix/vector demos + gradient descent
- `/optimizers` - 4-optimizer comparison
- `/playground` - 5 ML algorithms
- `/courses` - Placeholder (empty)

### Components by File Size
| File | LOC | Purpose |
|------|-----|---------|
| showcase.rs | 1,037 | **SPLIT THIS** - Matrix, vector, gradient descent demos |
| optimizer_demo.rs | 664 | 4 optimizers on 6 loss functions |
| ml_playground.rs | 495 | 5 algorithms (KMeans, PCA, LogReg, Scalers) |
| loss_functions.rs | 360 | Test functions for optimizers |
| correlation_heatmap.rs | 341 | N×N feature correlations |
| linear_regression_visualizer.rs | 311 | Tabbed viz (coefficients, importance, correlations) |
| coefficient_display.rs | 223 | Weights table + equation |
| csv_upload.rs | 208 | File upload + preview |
| feature_importance.rs | 192 | Importance bar chart |

**Total:** 3,949 lines across 12 components

---

## Missing Features Checklist

### P1: Critical Gaps (Week 1-2)
- [ ] WASM panic boundaries (silent crashes)
- [ ] Replace 62 `.unwrap()` calls
- [ ] Interactive parameter controls (ALL hardcoded now)
- [ ] Progress indicators (long-running tasks)
- [ ] Input validation (CSV limits, param ranges)

### P2: Educational (Week 3-4)
- [ ] Step-through debugging (watch algorithms iterate)
- [ ] Data exploration (summary stats, distributions)
- [ ] Scatter plot matrix (feature relationships)
- [ ] In-context tooltips (technical terms)
- [ ] Algorithm comparison mode (side-by-side)

### P3: Revolutionary (Month 2+)
- [ ] Pipeline builder (drag-drop chains)
- [ ] 3D loss surface visualization (WebGL)
- [ ] Guided tutorials (step-by-step walkthroughs)
- [ ] Performance benchmarks (Rust vs Python)
- [ ] Dark mode + accessibility

---

## Key File Paths

### Components
```
/Users/brunodossantos/Code/brunoml/cargo_workspace/web/src/components/
├─ ml_playground.rs     ← Add AlgorithmConfig here
├─ showcase.rs          ← SPLIT into vector_demos.rs, matrix_demos.rs, gradient_descent_demo.rs
├─ optimizer_demo.rs    ← Add 3D toggle
└─ (new files below)
```

### Create These Components
```
src/components/
├─ progress_bar.rs              # Show algorithm progress
├─ algorithm_config.rs          # Parameter sliders/inputs
├─ error_boundary.rs            # Catch WASM panics
├─ data_profiler.rs             # Dataset statistics
├─ scatter_matrix.rs            # N×N feature pairs
├─ algorithm_stepper.rs         # Play/Pause/Step controls
├─ comparison_arena.rs          # Side-by-side algorithms
├─ pipeline_builder.rs          # Drag-drop ML pipeline
├─ loss_surface_3d.rs           # WebGL visualization
└─ tooltip.rs                   # Hover help
```

### Traits to Extend
```
ml_traits/src/
├─ debuggable.rs                # NEW: step(), current_state(), reset()
├─ progress.rs                  # NEW: on_progress(), report_progress()
└─ error.rs                     # NEW: MLError enum (replace String)
```

---

## Algorithm Parameter Reference

### Currently Hardcoded (ml_playground.rs)

```rust
// Line 363
fn run_kmeans(dataset: &CsvDataset) -> String {
    let k = 3;  // ← ADD SLIDER (2-10)
    let mut kmeans = KMeans::new(k, 100, 1e-4, Some(42));
    //                          ^   ^    ^      ^
    //                          k   iter tol    seed
}

// Line 396
fn run_pca(dataset: &CsvDataset) -> String {
    let n_components = 2.min(dataset.features.cols);  // ← ADD SLIDER (1-N)
    let mut pca = PCA::new(n_components);
}

// Line 422
fn run_logistic_regression(dataset: &CsvDataset) -> String {
    let mut model = LogisticRegression::new(0.01, 1000, 1e-4);
    //                                     ^lr  ^iter ^tol
    // ← ADD INPUTS for learning_rate, max_iterations, tolerance
}
```

### Target UI
```
┌─────────────────────────────────────┐
│  Algorithm: K-Means                 │
│  ┌───────────────────────────────┐ │
│  │ Number of Clusters (k)        │ │
│  │ [========o==] k=5             │ │
│  │ Range: 2-10                   │ │
│  ├───────────────────────────────┤ │
│  │ Max Iterations                │ │
│  │ [500]                         │ │
│  ├───────────────────────────────┤ │
│  │ Tolerance                     │ │
│  │ [0.0001]                      │ │
│  └───────────────────────────────┘ │
│  [▶ Run Algorithm]                 │
└─────────────────────────────────────┘
```

---

## Performance Patterns (KEEP THESE)

### Zero-Allocation (optimizer_demo.rs:86)
```rust
// GOOD: 10-50× faster
self.position = self.optimizer.step_2d((x, y), (dx, dy));

// BAD: Creates 4 allocations per iteration
let weights = Matrix::from_vec(vec![x, y], 1, 2)?;
let gradient = Matrix::from_vec(vec![dx, dy], 1, 2)?;
optimizer.update_weights(0, &gradient, &mut weights, &shapes);
```

### Bounded Memory (optimizer_demo.rs:88-95)
```rust
const MAX_PATH_LENGTH: usize = 1000;
if self.path.len() >= MAX_PATH_LENGTH {
    self.path.remove(0);  // Circular buffer
}
self.path.push(self.position);
```

### Async Non-Blocking (ml_playground.rs:157)
```rust
onclick: move |_| {
    spawn(async move {
        is_processing.set(true);
        let result = run_algorithm(...);
        result_message.set(result);
        is_processing.set(false);
    });
}
```

---

## State Management Pattern

### Component-Local Signals
```rust
// Current approach (works well)
let mut csv_dataset = use_signal(|| None::<CsvDataset>);
let mut selected_algorithm = use_signal(|| Algorithm::KMeans);
let mut result_message = use_signal(|| String::new());
```

### Reactive Computations
```rust
// Auto-updates when inputs change
let result = use_memo(move || {
    let a = matrix_a();
    let b = matrix_b();
    a + b
});
```

### Global State (Proposed for large refactor)
```rust
// src/state.rs
pub struct AppState {
    pub dataset: Option<CsvDataset>,
    pub theme: Theme,
}

pub fn use_app_state() -> Signal<AppState> {
    use_context()
}
```

---

## Quick Implementation Snippets

### Add Panic Boundary
```rust
// Wrap algorithm execution in ml_playground.rs
use std::panic;

onclick: move |_| {
    spawn(async move {
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            run_algorithm(*selected_algorithm.read(), dataset)
        }));
        match result {
            Ok(msg) => result_message.set(msg),
            Err(_) => {
                result_message.set("❌ Algorithm crashed. Please reload and try simpler data.".to_string());
                web_sys::console::error_1(&"WASM panic caught".into());
            }
        }
    });
}
```

### Add Progress Indicator
```rust
// In algorithm runners
let mut progress = use_signal(|| 0);

for iter in 0..max_iterations {
    if iter % 10 == 0 {
        progress.set((iter * 100) / max_iterations);
        // Yield to browser
        gloo::timers::future::sleep(Duration::from_millis(0)).await;
    }
    // ... algorithm logic
}

// In UI
if *is_processing.read() {
    div { class: "progress-container",
        div { class: "progress-bar", style: "width: {progress}%" }
        p { "Processing: {progress}% complete" }
    }
}
```

### Add Parameter Control
```rust
// In ml_playground.rs
let mut kmeans_k = use_signal(|| 3);

div { class: "algorithm-config",
    label { "Number of Clusters (k):" }
    input {
        r#type: "range",
        min: "2",
        max: "10",
        value: "{kmeans_k}",
        oninput: move |evt| {
            kmeans_k.set(evt.value().parse().unwrap_or(3));
        }
    }
    span { "k = {kmeans_k}" }
}

// Use in runner
fn run_kmeans(dataset: &CsvDataset, k: usize) -> String {
    let mut kmeans = KMeans::new(k, 100, 1e-4, Some(42));
    // ...
}
```

---

## CSS Architecture

### Current: Single 1,880-line file
```
web/assets/main.css
├─ Lines 1-17: Base styles
├─ Lines 18-152: Showcase
├─ Lines 156-752: Gradient Descent
├─ Lines 754-1530: Linear Regression Viz
└─ Lines 1532-1880: ML Playground
```

### Proposed: Modular Structure
```
assets/css/
├─ base.css               # Reset, typography
├─ layout.css             # Grid, responsive
├─ components/
│   ├─ buttons.css
│   ├─ inputs.css
│   ├─ tables.css
│   └─ tooltips.css
├─ pages/
│   ├─ showcase.css
│   ├─ optimizer.css
│   └─ playground.css
└─ themes/
    ├─ light.css
    └─ dark.css
```

### Color Palette
```css
/* Primary */
--purple-gradient-start: #667eea;
--purple-gradient-end: #764ba2;

/* Accents */
--blue: #2563eb;        /* Data, regression lines */
--green: #10b981;       /* Success, positive coefficients */
--red: #ef4444;         /* Errors, negative coefficients */
--yellow: #f59e0b;      /* Warnings, highlights */

/* Neutrals */
--gray-50: #f9fafb;
--gray-800: #1f2937;
```

---

## Testing Commands

### Run Playwright E2E Tests
```bash
cd web
npm test                          # All tests
npm test -- routes.spec.js        # Specific test
npm test -- --headed              # Show browser
npm test -- --debug               # Debug mode
```

### Build & Serve
```bash
cd web
dx serve                          # Dev server (hot reload)
dx serve --hot-reload             # Faster reloads
dx build --platform web           # Production build
dx serve --platform desktop       # Native app
```

### Verify WASM Bundle Size
```bash
cd web
dx build --platform web --release
ls -lh target/dx/web/release/web/public/wasm-bindgen/*.wasm
```

---

## Common Patterns in Existing Components

### File Upload Pattern (csv_upload.rs:23-81)
```rust
let handle_upload = move |evt: Event<FormData>| async move {
    loading.set(true);
    error_message.set(None);

    if let Some(file_engine) = evt.files() {
        let files = file_engine.files();
        if files.is_empty() { /* error */ return; }

        // Validate extension
        if !filename.ends_with(".csv") { /* error */ return; }

        // Read file
        if let Some(contents) = file_engine.read_file_to_string(&filename).await {
            // Parse & store
            file_content.set(Some(contents));
        }
    }
    loading.set(false);
};
```

### Tabbed Interface Pattern (linear_regression_visualizer.rs:57-73)
```rust
let mut active_tab = use_signal(|| "coefficients");

// Tab buttons
button {
    class: if active_tab() == "coefficients" { "tab-button active" } else { "tab-button" },
    onclick: move |_| active_tab.set("coefficients"),
    "📋 Coefficients"
}

// Conditional rendering
if active_tab() == "coefficients" {
    div { class: "tab-panel",
        CoefficientDisplay { ... }
    }
}
```

### SVG Visualization Pattern (showcase.rs:912-1012)
```rust
svg {
    view_box: "0 0 100 100",
    xmlns: "http://www.w3.org/2000/svg",

    // Helper function
    let to_svg_x = |x: f64| -> f64 {
        (x - plot_x_min) / plot_width * 100.0
    };

    // Elements
    g { class: "data-points",
        for (x, y) in data.clone() {
            circle {
                cx: "{to_svg_x(x)}",
                cy: "{to_svg_y(y)}",
                r: "2",
                fill: "#10b981"
            }
        }
    }
}
```

---

## Error Handling Evolution

### Current (String-based)
```rust
match kmeans.fit(&dataset.features) {
    Ok(_) => "✅ Success",
    Err(e) => format!("❌ Failed: {}", e),
}
```

### Proposed (Structured MLError)
```rust
// ml_traits/src/error.rs
#[derive(Debug, Clone)]
pub enum MLError {
    InvalidInput { message: String, parameter: &'static str },
    NotFitted { model_type: &'static str },
    DimensionMismatch { expected: (usize, usize), got: (usize, usize) },
    ConvergenceFailure { iterations: usize, final_cost: f64 },
}

// Usage
match kmeans.fit(&dataset.features) {
    Ok(_) => "✅ Success",
    Err(MLError::DimensionMismatch { expected, got }) => {
        format!("❌ Shape mismatch: expected {:?}, got {:?}", expected, got)
    }
    Err(e) => format!("❌ Error: {:?}", e),
}
```

---

## Visualization Performance Targets

| Visualization | Current | Target | Optimization |
|---------------|---------|--------|--------------|
| Scatter Plot (2D) | SVG | 60 FPS | Canvas for >50 points |
| Heatmap (50×50) | 2500 SVG rects | 30 FPS | Pre-render to image |
| 3D Loss Surface | N/A | 30+ FPS | WebGL LOD mesh |
| Scatter Matrix (10×10) | N/A | <1s load | Lazy render (viewport only) |
| Algorithm Path | SVG polyline | 60 FPS | Bounded to 1000 points |

---

## Trait Implementation Checklist

### DebuggableModel (New Trait)
```rust
// Implement for each algorithm
impl DebuggableModel<f64, Matrix<f64>> for KMeans {
    type State = KMeansState;

    fn fit_initial(&mut self, X: &Matrix<f64>) -> Result<(), MLError> { ... }
    fn step(&mut self) -> Result<Self::State, MLError> { ... }
    fn current_state(&self) -> Self::State { ... }
    fn reset(&mut self) { ... }
    fn is_converged(&self) -> bool { ... }
    fn iteration_count(&self) -> usize { ... }
}

// Priority order:
// 1. KMeans (easiest to visualize - centroids + assignments)
// 2. PCA (eigenvector convergence)
// 3. LogisticRegression (decision boundary evolution)
```

---

## Quick Wins (1-2 hours each)

1. **Add Loading Spinners** - Show during algorithm execution
2. **Display Algorithm Runtime** - `performance.now()` before/after
3. **Show Dataset Shape** - Display (N samples, M features) prominently
4. **Hyperparameter Tooltips** - Explain what each param does
5. **Copy Model Equation** - Already implemented for LinReg, extend to others
6. **Dark Mode Toggle** - CSS variables + localStorage
7. **Keyboard Shortcuts** - Space = play/pause, arrows = step back/forward
8. **Export Results as JSON** - Download button for predictions/stats
9. **Sample Dataset Buttons** - Preload iris.csv, boston.csv, etc.
10. **Error Message Improvements** - Add "How to Fix" suggestions

---

## Resources

### Documentation
- Main analysis: `/docs/WEB_UI_ARCHITECTURE_ANALYSIS.md` (89KB)
- CLAUDE.md: Project architecture & patterns
- Dioxus docs: `web/DIOXUS_QUICK_REFERENCE.md`

### Examples
- Vector demo: `showcase.rs:34-65`
- Matrix ops: `showcase.rs:203-474`
- CSV upload: `csv_upload.rs:23-81`
- Tabbed UI: `linear_regression_visualizer.rs:57-104`
- SVG viz: `showcase.rs:912-1012`

### Testing
- E2E tests: `/web/tests/*.spec.js`
- Testing guide: `/web/TESTING.md`

---

**Last Updated:** November 8, 2025
**Status:** Ready for UI overhaul implementation
**Priority:** Start with P1 (Safety & Parameters) before adding new features
