# Dioxus 0.6.0 Capabilities for ML Visualizations

## Table of Contents

1. [Overview](#overview)
2. [Reactivity System](#reactivity-system)
3. [State Management](#state-management)
4. [Visualization Capabilities](#visualization-capabilities)
5. [Animation & Interaction](#animation--interaction)
6. [Performance & Optimization](#performance--optimization)
7. [Integration with Rust Libraries](#integration-with-rust-libraries)
8. [Best Practices for ML Visualizations](#best-practices-for-ml-visualizations)

---

## Overview

Dioxus 0.6.0 is a fullstack, cross-platform app framework for Rust that supports web, desktop, and mobile platforms from a single codebase. It provides fine-grained reactivity similar to SolidJS/Svelte and leverages Rust's performance for building sophisticated, educational ML visualizations.

### Key Strengths
- **Performance**: Benchmarks show Dioxus is the 3rd fastest framework (via Sledgehammer), ahead of React, Vue, and comparable to SolidJS
- **Type Safety**: Full Rust type system for compile-time correctness
- **Cross-Platform**: Single codebase for web (WASM), desktop, and mobile
- **Bundle Size**: Optimized builds can achieve <100kb for simple apps, ~234kb for complex apps

---

## Reactivity System

### Signals: Core Reactive Primitive

Signals are Copy state management primitives with automatic dependency tracking. Components only subscribe when they read the signal.

```rust
use dioxus::prelude::*;

fn App() -> Element {
    let mut count = use_signal(|| 0);

    rsx! {
        h1 { "Count: {count}" }
        button {
            onclick: move |_| count += 1,
            "Increment"
        }
    }
}
```

**Key Characteristics:**
- **Copy semantics**: Signals implement `Copy` even if inner value doesn't
- **Automatic tracking**: Reading a signal subscribes the component
- **Selective updates**: Only components that read the signal re-render
- **No subscriptions in futures**: Reading in async or event handlers doesn't subscribe

### Memos: Derived State

`use_memo` creates computed values that automatically update when dependencies change:

```rust
fn App() -> Element {
    let mut count = use_signal(|| 0);
    let double = use_memo(move || count() * 2);
    let squared = use_memo(move || count() * count());

    rsx! {
        div {
            p { "Count: {count}" }
            p { "Double: {double}" }
            p { "Squared: {squared}" }
            button { onclick: move |_| count += 1, "Increment" }
        }
    }
}
```

**Performance Benefits:**
- Only recomputes when dependencies change
- Memoized results prevent unnecessary work
- Multiple derived values can form computation graphs

### Resources: Async State

`use_resource` manages async operations with reactive dependencies:

```rust
fn WeatherApp() -> Element {
    let mut city = use_signal(|| "San Francisco".to_string());

    let weather = use_resource(move || async move {
        fetch_weather(&city()).await
    });

    rsx! {
        match &*weather.read() {
            Some(Ok(data)) => rsx! { WeatherDisplay { data } },
            Some(Err(e)) => rsx! { p { "Error: {e}" } },
            None => rsx! { p { "Loading..." } }
        }
    }
}
```

**Use Cases:**
- API calls that depend on reactive state
- Real-time data fetching
- Progressive data loading for visualizations

### Effects: Side Effects

`use_effect` runs reactive closures after component rendering:

```rust
fn App() -> Element {
    let mut count = use_signal(|| 0);

    use_effect(move || {
        // Runs when count changes
        println!("Count changed to: {}", count());
    });

    rsx! {
        button { onclick: move |_| count += 1, "Click me" }
    }
}
```

**Common Use Cases:**
- DOM manipulation after render
- Analytics tracking
- Synchronizing with external systems
- Logging state changes

---

## State Management

### Local State: Props

Best for component-specific state:

```rust
#[component]
fn Chart(data: Signal<Vec<f64>>, width: i32, height: i32) -> Element {
    rsx! {
        svg { width: "{width}", height: "{height}",
            // Render chart
        }
    }
}
```

### Shared State: Context API

Avoid prop drilling with context:

```rust
// Provider (parent component)
fn App() -> Element {
    use_context_provider(|| Signal::new(ModelState::default()));

    rsx! {
        Visualization {}
        Controls {}
    }
}

// Consumer (child component)
fn Controls() -> Element {
    let mut model = use_context::<Signal<ModelState>>();

    rsx! {
        button {
            onclick: move |_| model.write().train(),
            "Train Model"
        }
    }
}
```

**Characteristics:**
- Scoped to component subtree
- Immutable after creation (use signals inside for mutability)
- No prop drilling
- Component reuse maintains separate state

### Global State: Global Signals

Application-wide state with Rust statics:

```rust
static TRAINING_DATA: GlobalSignal<Vec<DataPoint>> = Signal::global(Vec::new);

fn anywhere_in_app() -> Element {
    let data = TRAINING_DATA();

    rsx! {
        p { "Dataset size: {data.len()}" }
    }
}
```

### Coroutines: Long-Running Tasks

`use_coroutine` manages background tasks with message passing:

```rust
enum TrainingAction {
    Start,
    Pause,
    SetLearningRate(f64),
}

fn TrainingControls() -> Element {
    let trainer = use_coroutine(|mut rx: UnboundedReceiver<TrainingAction>| async move {
        let mut model = NeuralNetwork::new();

        while let Some(action) = rx.next().await {
            match action {
                TrainingAction::Start => {
                    // Run training loop
                    for epoch in 0..100 {
                        model.train_epoch();
                        // Update visualization state
                    }
                }
                TrainingAction::Pause => break,
                TrainingAction::SetLearningRate(lr) => model.set_lr(lr),
            }
        }
    });

    rsx! {
        button {
            onclick: move |_| trainer.send(TrainingAction::Start),
            "Start Training"
        }
    }
}
```

---

## Visualization Capabilities

### SVG Rendering

Native SVG support for vector graphics:

```rust
fn LinearRegressionPlot(data: Vec<(f64, f64)>, line: (f64, f64)) -> Element {
    let (slope, intercept) = line;

    rsx! {
        svg {
            width: "600",
            height: "400",
            view_box: "0 0 600 400",

            // Axes
            line { x1: "50", y1: "350", x2: "550", y2: "350", stroke: "black" }
            line { x1: "50", y1: "50", x2: "50", y2: "350", stroke: "black" }

            // Data points
            for (x, y) in data.iter() {
                circle {
                    cx: "{x * 5.0 + 50.0}",
                    cy: "{350.0 - y * 5.0}",
                    r: "3",
                    fill: "blue"
                }
            }

            // Regression line
            line {
                x1: "50",
                y1: "{350.0 - (slope * 0.0 + intercept) * 5.0}",
                x2: "550",
                y2: "{350.0 - (slope * 100.0 + intercept) * 5.0}",
                stroke: "red",
                stroke_width: "2"
            }
        }
    }
}
```

### dioxus-charts Library

Simple, CSS-customizable chart components:

```rust
use dioxus_charts::{BarChart, LineChart, PieChart};

fn ChartExample() -> Element {
    rsx! {
        LineChart {
            padding_top: 30,
            padding_left: 70,
            padding_right: 50,
            padding_bottom: 30,
            series: vec![
                vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
                vec![2.0, 4.0, 3.0, 6.0, 5.0, 7.0]
            ],
            labels: vec!["Jan".into(), "Feb".into(), "Mar".into(),
                         "Apr".into(), "May".into(), "Jun".into()]
        }

        BarChart {
            bar_width: "10%",
            horizontal_bars: true,
            label_interpolation: (|v| format!("{v}%")) as fn(f32) -> String,
            series: vec![vec![63.0, 14.4, 8.0, 5.1, 1.8]],
            labels: vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()]
        }
    }
}
```

**Supported Chart Types:**
- Line charts
- Bar charts (vertical/horizontal)
- Stacked bar charts
- Pie charts
- Donut charts
- Gauge visualizations

**Installation:**
```toml
[dependencies]
dioxus-charts = "0.3"
```

### plotters-dioxus Integration

Use the Plotters library for complex scientific visualizations:

```rust
// plotters-dioxus provides a Dioxus backend for Plotters
// Enables advanced plotting capabilities within Dioxus

// Installation:
// [dependencies]
// plotters-dioxus = "0.1"
```

**Capabilities:**
- Statistical plots
- Heat maps
- 3D surface plots
- Custom drawing backends
- Full Plotters API

### Canvas Support

HTML5 Canvas for pixel-based rendering:

```rust
fn CanvasVisualization() -> Element {
    let canvas_ref = use_signal(|| None::<web_sys::HtmlCanvasElement>);

    use_effect(move || {
        if let Some(canvas) = canvas_ref() {
            let ctx = canvas
                .get_context("2d")
                .unwrap()
                .unwrap()
                .dyn_into::<web_sys::CanvasRenderingContext2d>()
                .unwrap();

            // Draw on canvas
            ctx.begin_path();
            ctx.arc(75.0, 75.0, 50.0, 0.0, std::f64::consts::PI * 2.0).unwrap();
            ctx.stroke();
        }
    });

    rsx! {
        canvas {
            width: "800",
            height: "600",
            onmounted: move |evt| {
                canvas_ref.set(Some(evt.data.downcast::<web_sys::HtmlCanvasElement>()));
            }
        }
    }
}
```

### WebGL/3D Graphics

Limited support currently, with plans for improved integration:

- **Current**: Can use `wgpu` for 3D rendering, but it occupies entire window
- **Future**: Dioxus plans to move to Blitz (custom renderer with WGPU integration)
- **Workaround**: Use `web-sys` to access WebGL directly via JavaScript interop

---

## Animation & Interaction

### Animation Libraries

#### dioxus-motion

Lightweight, cross-platform animation library:

```rust
// Installation:
// [dependencies]
// dioxus-motion = "0.1"

// Features:
// - Spring animations
// - CSS-like transitions
// - Timeline-based animations
// - Cross-platform (web, desktop, mobile)
```

**Note**: Platform-agnostic timing system with fallbacks for different browsers

#### dioxus-spring

Physics-based spring animations:

```rust
// Installation:
// [dependencies]
// dioxus-spring = "0.1"

// Pairs well with dioxus-use-gesture for interactive animations
```

### Animation Patterns

#### CSS Transitions

```rust
fn AnimatedComponent() -> Element {
    let mut visible = use_signal(|| false);

    rsx! {
        div {
            class: if visible() { "fade-in" } else { "fade-out" },
            "Animated content"
        }
        button {
            onclick: move |_| visible.set(!visible()),
            "Toggle"
        }
    }
}
```

```css
/* assets/main.css */
.fade-in {
    opacity: 1;
    transition: opacity 0.3s ease-in-out;
}

.fade-out {
    opacity: 0;
    transition: opacity 0.3s ease-in-out;
}
```

#### Programmatic Animations

Using JavaScript interop for requestAnimationFrame:

```rust
fn AnimatedPlot() -> Element {
    let mut progress = use_signal(|| 0.0);

    use_coroutine(move |_: UnboundedReceiver<()>| async move {
        loop {
            // Use async sleep to yield to Dioxus scheduler
            // (requestAnimationFrame would block single-threaded runtime)
            tokio::time::sleep(Duration::from_millis(16)).await;

            progress.set((progress() + 0.01) % 1.0);
        }
    });

    rsx! {
        svg { width: "400", height: "400",
            circle {
                cx: "{200.0 + progress() * 100.0}",
                cy: "200",
                r: "10",
                fill: "blue"
            }
        }
    }
}
```

**Important**: Use `tokio::time::sleep` or similar async yields instead of tight loops to prevent blocking the Dioxus renderer.

### Event Handlers

#### Mouse Events

```rust
fn InteractivePlot() -> Element {
    let mut mouse_pos = use_signal(|| (0.0, 0.0));
    let mut dragging = use_signal(|| false);

    rsx! {
        svg {
            width: "600",
            height: "400",
            onmousedown: move |_| dragging.set(true),
            onmouseup: move |_| dragging.set(false),
            onmousemove: move |evt| {
                if dragging() {
                    let x = evt.page_coordinates().x;
                    let y = evt.page_coordinates().y;
                    mouse_pos.set((x, y));
                }
            },

            circle {
                cx: "{mouse_pos().0}",
                cy: "{mouse_pos().1}",
                r: "10",
                fill: "red"
            }
        }
    }
}
```

#### Keyboard Events

```rust
fn KeyboardControls() -> Element {
    let mut step = use_signal(|| 0);

    rsx! {
        div {
            tabindex: 0,
            onkeydown: move |evt| {
                match evt.key().as_str() {
                    "ArrowRight" => step += 1,
                    "ArrowLeft" => step -= 1,
                    _ => {}
                }
            },
            "Step: {step} (use arrow keys)"
        }
    }
}
```

#### Touch Events

```rust
fn TouchInteractive() -> Element {
    let mut touch_active = use_signal(|| false);

    rsx! {
        div {
            ontouchstart: move |_| touch_active.set(true),
            ontouchend: move |_| touch_active.set(false),
            class: if touch_active() { "touched" } else { "" },
            "Touch me"
        }
    }
}
```

### Drag and Drop

```rust
fn DragDropExample() -> Element {
    let mut drag_data = use_signal(|| None::<String>);

    rsx! {
        div {
            draggable: true,
            ondragstart: move |evt| {
                evt.set_data("text/plain", "Data being dragged");
            },
            "Drag me"
        }

        div {
            ondrop: move |evt| {
                evt.prevent_default();
                let data = evt.data("text/plain");
                drag_data.set(Some(data));
            },
            ondragover: move |evt| {
                evt.prevent_default();
            },
            "Drop zone: {drag_data():?}"
        }
    }
}
```

---

## Performance & Optimization

### WASM Bundle Size Optimization

#### Cargo Profile (Stable)

```toml
[profile.release]
opt-level = "z"          # Optimize for size
lto = true               # Enable link-time optimization
codegen-units = 1        # Reduce parallel codegen for better optimization
strip = true             # Strip symbols
panic = "abort"          # Reduce panic handling code
```

**Result**: Reduces bundle from ~2.36MB to ~310KB

#### wasm-opt (Recommended)

```bash
# Install binaryen
cargo install wasm-opt

# Optimize WASM file
wasm-opt -Oz -o optimized.wasm input.wasm
```

**Result**: Additional ~100KB reduction (310KB → 234KB)

#### Nightly Features (Unstable)

```toml
[unstable]
build-std = ["std", "panic_abort"]
build-std-features = ["panic_immediate_abort"]
```

**Result**: Further reduction to <100KB for simple apps

### Runtime Performance

#### Minimize Re-renders

```rust
// BAD: Entire component re-renders on any data change
fn App() -> Element {
    let mut data = use_signal(|| vec![0; 1000]);

    rsx! {
        for item in data() {
            Item { value: item }
        }
    }
}

// GOOD: Only changed items re-render
fn App() -> Element {
    let mut data = use_signal(|| vec![0; 1000]);

    rsx! {
        for (idx, item) in data().iter().enumerate() {
            Item { key: "{idx}", value: *item }
        }
    }
}
```

#### Use Memos for Expensive Computations

```rust
fn MLVisualization() -> Element {
    let mut weights = use_signal(|| vec![0.5; 100]);

    // Expensive computation only runs when weights change
    let predictions = use_memo(move || {
        compute_predictions(&weights())  // Heavy computation
    });

    rsx! {
        PredictionPlot { data: predictions() }
    }
}
```

#### Batch State Updates

```rust
// BAD: Multiple separate updates
fn update_model() {
    weights.set(new_weights);      // Re-render
    bias.set(new_bias);            // Re-render
    loss.set(new_loss);            // Re-render
}

// GOOD: Single atomic update
fn update_model() {
    model_state.write().update(new_weights, new_bias, new_loss);  // Single re-render
}
```

#### Avoid Holding Borrows

```rust
// BAD: Panics at runtime
let read = signal.read();
signal += 1;  // PANIC: overlapping borrows

// GOOD: Use scoped blocks
{
    let value = signal.read();
    // Use value
}
signal += 1;  // OK

// BETTER: Use .with() helper
signal.with(|value| {
    // Use value
});
signal += 1;  // OK
```

### Large Lists

**Current Limitations:**
- Safari performance issues with >2000 items
- No built-in virtualization (as of 0.6.0)

**Workarounds:**
- Implement windowing manually
- Use pagination
- Progressive rendering with `use_resource`

```rust
fn VirtualizedList() -> Element {
    let mut visible_range = use_signal(|| 0..100);
    let data = use_signal(|| (0..10000).collect::<Vec<_>>());

    rsx! {
        div {
            onscroll: move |evt| {
                // Update visible range based on scroll position
                let scroll_top = evt.scroll_top();
                let item_height = 50;
                let start = (scroll_top / item_height) as usize;
                visible_range.set(start..start + 100);
            },

            for item in data()[visible_range()].iter() {
                Item { value: *item }
            }
        }
    }
}
```

---

## Integration with Rust Libraries

### Direct Integration

Rust algorithms run natively in WASM with near-native performance:

```rust
use linear_algebra::{Matrix, Vector};
use linear_regression::LinearRegression;

fn MLTrainingComponent() -> Element {
    let mut model = use_signal(|| LinearRegression::new());
    let training_data = use_signal(|| load_training_data());

    let train = move |_| {
        let mut m = model.write();
        m.fit(&training_data());
    };

    rsx! {
        button { onclick: train, "Train Model" }
        ModelVisualization { model: model() }
    }
}
```

### Async Computation

Prevent UI blocking with async tasks:

```rust
fn ExpensiveComputation() -> Element {
    let mut result = use_signal(|| None);

    let compute = use_resource(move || async move {
        // Run heavy computation in async task
        tokio::task::spawn_blocking(|| {
            expensive_ml_algorithm()
        }).await
    });

    rsx! {
        match &*compute.read() {
            Some(Ok(data)) => rsx! { ResultDisplay { data } },
            Some(Err(e)) => rsx! { p { "Error: {e}" } },
            None => rsx! { p { "Computing..." } }
        }
    }
}
```

### JavaScript Interop

Access browser APIs and JavaScript libraries:

```rust
fn ChartWithD3() -> Element {
    use_effect(|| {
        eval(r#"
            // Use D3.js or other JavaScript libraries
            d3.select("#my-chart")
              .append("svg")
              .attr("width", 600)
              .attr("height", 400);
        "#);
    });

    rsx! {
        div { id: "my-chart" }
    }
}
```

#### use_eval Hook

Bidirectional Rust-JavaScript communication:

```rust
fn JSInterop() -> Element {
    let mut eval_result = use_signal(|| String::new());

    let run_js = move |_| {
        let eval = eval(r#"
            // JavaScript can send to Rust
            dioxus.send("Hello from JS!");

            // JavaScript can receive from Rust
            const msg = await dioxus.recv();
            console.log(msg);

            return "Computation complete";
        "#);

        spawn(async move {
            // Rust sends to JavaScript
            eval.send("Hello from Rust!").await;

            // Rust receives from JavaScript
            if let Ok(msg) = eval.recv().await {
                eval_result.set(msg.to_string());
            }
        });
    };

    rsx! {
        button { onclick: run_js, "Run JS" }
        p { "{eval_result}" }
    }
}
```

#### wasm-bindgen Integration

Direct access to web APIs:

```rust
use wasm_bindgen::prelude::*;
use web_sys::{console, Performance};

#[wasm_bindgen]
extern "C" {
    fn customJSFunction(x: i32) -> i32;
}

fn WebAPIs() -> Element {
    let performance = web_sys::window()
        .unwrap()
        .performance()
        .unwrap();

    let start = performance.now();

    // Your code

    let duration = performance.now() - start;
    console::log_1(&format!("Took {duration}ms").into());

    rsx! {
        p { "Check browser console" }
    }
}
```

### Web Workers (Future)

Not currently built-in, but can use with `wasm-bindgen`:

```rust
// Requires manual setup with wasm-bindgen and web-sys
// No official Dioxus integration yet
```

---

## Best Practices for ML Visualizations

### 1. Reactive Data Flow

```rust
// Training data → Model → Predictions → Visualization
static TRAINING_DATA: GlobalSignal<Vec<DataPoint>> = Signal::global(Vec::new);

fn MLPipeline() -> Element {
    let model = use_memo(move || {
        train_model(&TRAINING_DATA())
    });

    let predictions = use_memo(move || {
        model().predict_all(&TRAINING_DATA())
    });

    rsx! {
        DataInput {}
        ModelControls {}
        PredictionPlot {
            actual: TRAINING_DATA(),
            predicted: predictions()
        }
    }
}
```

### 2. Progressive Rendering

Load and display data incrementally:

```rust
fn LargeDatasetViz() -> Element {
    let loaded_count = use_signal(|| 0);
    let data = use_signal(Vec::<DataPoint>::new);

    use_resource(move || async move {
        for chunk in dataset_chunks() {
            data.write().extend(chunk);
            loaded_count += chunk.len();

            // Yield to renderer
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    rsx! {
        p { "Loaded {loaded_count()} points" }
        Scatterplot { data: data() }
    }
}
```

### 3. Interactive Training

Real-time visualization of model training:

```rust
enum TrainingMsg {
    Step,
    UpdateLearningRate(f64),
}

fn InteractiveTraining() -> Element {
    let mut model_state = use_signal(|| ModelState::new());

    let trainer = use_coroutine(move |mut rx| async move {
        while let Some(msg) = rx.next().await {
            match msg {
                TrainingMsg::Step => {
                    model_state.write().train_step();
                }
                TrainingMsg::UpdateLearningRate(lr) => {
                    model_state.write().set_learning_rate(lr);
                }
            }

            // Yield to allow visualization updates
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    });

    rsx! {
        LossPlot { history: model_state().loss_history }
        WeightVisualization { weights: model_state().weights }

        button {
            onclick: move |_| trainer.send(TrainingMsg::Step),
            "Train Step"
        }
    }
}
```

### 4. Modular Visualization Components

```rust
#[component]
fn Axis(
    start: (f64, f64),
    end: (f64, f64),
    label: String,
    ticks: Vec<(f64, String)>
) -> Element {
    rsx! {
        g { class: "axis",
            line {
                x1: "{start.0}", y1: "{start.1}",
                x2: "{end.0}", y2: "{end.1}",
                stroke: "black"
            }

            for (pos, text) in ticks {
                text {
                    x: "{pos}", y: "{end.1 + 15}",
                    "{text}"
                }
            }
        }
    }
}

#[component]
fn Scatterplot(data: Vec<(f64, f64)>, color: String) -> Element {
    rsx! {
        g { class: "scatterplot",
            for (x, y) in data {
                circle { cx: "{x}", cy: "{y}", r: "3", fill: "{color}" }
            }
        }
    }
}

#[component]
fn MLVisualization() -> Element {
    rsx! {
        svg { width: "800", height: "600", view_box: "0 0 800 600",
            Axis {
                start: (50.0, 550.0),
                end: (750.0, 550.0),
                label: "X".into(),
                ticks: vec![(50.0, "0".into()), (400.0, "50".into()), (750.0, "100".into())]
            }

            Axis {
                start: (50.0, 550.0),
                end: (50.0, 50.0),
                label: "Y".into(),
                ticks: vec![(550.0, "0".into()), (300.0, "50".into()), (50.0, "100".into())]
            }

            Scatterplot { data: training_data(), color: "blue".into() }
            Scatterplot { data: predictions(), color: "red".into() }
        }
    }
}
```

### 5. Error Handling

```rust
fn RobustMLComponent() -> Element {
    let mut error = use_signal(|| None::<String>);

    let training_data = use_resource(move || async move {
        match load_data().await {
            Ok(data) => data,
            Err(e) => {
                error.set(Some(e.to_string()));
                vec![]
            }
        }
    });

    rsx! {
        if let Some(err) = error() {
            div { class: "error",
                p { "Error: {err}" }
                button {
                    onclick: move |_| error.set(None),
                    "Dismiss"
                }
            }
        }

        match &*training_data.read() {
            Some(data) if !data.is_empty() => rsx! {
                Visualization { data }
            },
            Some(_) => rsx! { p { "No data available" } },
            None => rsx! { p { "Loading..." } }
        }
    }
}
```

### 6. Responsive Design

```rust
fn ResponsiveChart() -> Element {
    let mut dimensions = use_signal(|| (800, 600));

    use_effect(move || {
        // Listen to window resize
        let closure = Closure::wrap(Box::new(move || {
            let window = web_sys::window().unwrap();
            let width = window.inner_width().unwrap().as_f64().unwrap() as i32;
            let height = window.inner_height().unwrap().as_f64().unwrap() as i32;
            dimensions.set((width * 0.8, height * 0.6));
        }) as Box<dyn FnMut()>);

        web_sys::window()
            .unwrap()
            .add_event_listener_with_callback("resize", closure.as_ref().unchecked_ref())
            .unwrap();

        closure.forget();
    });

    rsx! {
        svg {
            width: "{dimensions().0}",
            height: "{dimensions().1}",
            // Chart content
        }
    }
}
```

---

## Additional Resources

### Official Documentation
- [Dioxus Guide](https://dioxuslabs.com/learn/0.6/guide)
- [API Documentation](https://docs.rs/dioxus/latest/dioxus/)
- [Examples Repository](https://github.com/DioxusLabs/dioxus/tree/main/examples)

### Community Libraries
- [dioxus-charts](https://github.com/dioxus-community/dioxus-charts) - Simple chart components
- [plotters-dioxus](https://github.com/DorianPinaud/plotters-dioxus) - Plotters integration
- [dioxus-motion](https://github.com/wheregmis/dioxus-motion) - Animation library
- [dioxus-spring](https://github.com/dioxus-community/dioxus-spring) - Spring animations
- [dioxus-radio](https://github.com/dioxus-community/dioxus-radio) - Granular state management

### Tools
- [dx CLI](https://github.com/DioxusLabs/dioxus/tree/main/packages/cli) - Development server and bundler
- [wasm-opt](https://github.com/WebAssembly/binaryen) - WASM optimization
- [trunk](https://trunkrs.dev/) - Alternative WASM bundler

### Performance Benchmarks
- [JS Framework Benchmark](https://krausest.github.io/js-framework-benchmark/) - Includes Dioxus
- Dioxus ranks 3rd overall via Sledgehammer
- Ahead of React, Vue, Angular
- Competitive with SolidJS and Svelte

---

## Summary

Dioxus 0.6.0 provides a powerful foundation for building sophisticated ML visualizations:

**Strengths:**
- Fine-grained reactivity for efficient updates
- Multiple state management patterns (local, context, global)
- Strong SVG support with native rendering
- Good performance (3rd fastest framework)
- Type-safe integration with Rust ML libraries
- Cross-platform deployment

**Considerations:**
- Limited built-in animation support (use community libraries)
- No native virtualization for large lists
- WebGL/3D support still evolving
- WASM bundle sizes require optimization
- Some platform-specific workarounds needed

**Ideal For:**
- Educational ML visualizations
- Interactive algorithm demonstrations
- Real-time training visualization
- Data exploration tools
- Prototype ML applications

For sophisticated, performant, and educational ML visualizations, Dioxus 0.6.0 offers excellent capabilities when combined with thoughtful architecture and the patterns documented above.
