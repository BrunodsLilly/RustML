# Dioxus Quick Reference - ML Platform Patterns

**One-page reference for common Dioxus patterns used in ML education platform**

---

## State Management

```rust
// LOCAL STATE - Single component
let mut count = use_signal(|| 0);
count.set(42);              // Write
let value = count();        // Read (shorthand)
let value = *count.read();  // Read (explicit)

// SHARED STATE - Parent to children
#[derive(Clone, Copy)]
struct Config { theme: Signal<String> }

use_context_provider(|| Config { theme: Signal::new("dark") });
let config = use_context::<Config>();

// GLOBAL STATE - Everywhere
static CACHE: GlobalSignal<HashMap<String, Data>> = Signal::global(HashMap::new);
CACHE.write().insert("key", data);
```

---

## Routing

```rust
// DEFINE ROUTES
#[derive(Routable, Clone, PartialEq)]
enum Route {
    #[route("/")]
    Home,
    #[route("/playground/:algorithm")]
    Playground { algorithm: String },
    #[route("/config?:lr&:epochs")]
    Config { lr: Option<f64>, epochs: Option<usize> },
}

// NAVIGATE
let nav = use_navigator();
nav.push(Route::Playground { algorithm: "kmeans".to_string() });

// LINK
Link { to: Route::Home, "Home" }
```

---

## Async & Safety (CRITICAL FOR WASM)

```rust
// SPAWN ASYNC TASK
spawn(async move {
    let result = algorithm.fit(&data).await;
    state.set(result);
});

// PANIC RECOVERY (prevents WASM crash)
use std::panic;

let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
    run_algorithm(data)
}));

match result {
    Ok(val) => /* success */,
    Err(_) => console::error_1(&"Crashed!".into()),
}

// TIMEOUT (5 seconds)
use gloo::timers::future::TimeoutFuture;

async fn with_timeout<F, T>(future: F, ms: u32) -> Result<T, String>
where F: Future<Output = T> {
    // ... see full implementation in guide
}

// DEBOUNCE USER INPUT
let mut task = use_signal(|| None);
let task_handle = spawn(async move {
    TimeoutFuture::new(300).await;
    handle_input();
});
task.set(Some(task_handle));
```

---

## File Upload

```rust
// BASIC FILE UPLOAD
input {
    r#type: "file",
    accept: ".csv",
    onchange: move |evt| {
        spawn(async move {
            if let Some(file_engine) = evt.files() {
                let files = file_engine.files();
                if let Some(file_name) = files.first() {
                    let contents = file_engine.read_file(file_name).await;
                    // Process contents
                }
            }
        });
    }
}

// DRAG & DROP
div {
    ondrop: move |evt| {
        evt.prevent_default();
        // Process evt.files()
    },
    ondragover: move |evt| evt.prevent_default(),
}
```

---

## Forms & Inputs

```rust
// RANGE SLIDER
let mut value = use_signal(|| 3);

input {
    r#type: "range",
    min: "2",
    max: "10",
    value: "{value()}",
    oninput: move |evt| {
        value.set(evt.value().parse().unwrap_or(3));
    },
}
span { "k = {value()}" }

// VALIDATION
let mut error = use_signal(|| None::<String>);

input {
    oninput: move |evt| {
        let val = evt.value();
        if let Err(e) = validate(&val) {
            error.set(Some(e));
        } else {
            error.set(None);
        }
    },
    class: if error().is_some() { "invalid" } else { "" },
}
```

---

## SVG Visualization

```rust
// BASIC SVG
svg {
    width: "500",
    height: "500",
    viewBox: "0 0 500 500",

    // Rectangle
    rect {
        x: "10",
        y: "10",
        width: "100",
        height: "100",
        fill: "blue",
    }

    // Circle with interaction
    circle {
        cx: "{x}",
        cy: "{y}",
        r: "5",
        fill: "red",
        onmouseover: move |_| tooltip.set(Some(data)),
    }

    // Text label
    text {
        x: "50",
        y: "50",
        text_anchor: "middle",
        "Label"
    }
}

// OPTIMIZE: Use path instead of many circles
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

path {
    d: "{path_data}",
    stroke: "blue",
    fill: "none",
}
```

---

## Styling

```css
/* CSS CUSTOM PROPERTIES (main.css) */
:root {
  --primary-color: #4A90E2;
  --bg-primary: #FFFFFF;
  --transition-fast: 150ms ease;
}

[data-theme="dark"] {
  --primary-color: #6BA4E7;
  --bg-primary: #1E1E1E;
}

.card {
  background: var(--bg-primary);
  transition: transform var(--transition-fast);
}
```

```rust
// CONDITIONAL CLASSES
class: if active { "btn active" } else { "btn" },

// MULTIPLE CONDITIONS
class: format!(
    "card {} {}",
    if selected { "selected" } else { "" },
    if disabled { "disabled" } else { "" },
),

// DYNAMIC INLINE STYLES (avoid unless necessary)
style: "--color: {color}; --size: {size}px;",
```

---

## Performance

```rust
// PRE-COMPUTE BEFORE RENDER
let values: Vec<f64> = (0..1000)
    .map(|i| expensive_calc(i))
    .collect();

rsx! {
    for val in values { div { "{val}" } }
}

// MEMO FOR REACTIVE COMPUTATION
let computed = use_memo(move || {
    expensive_calc(*input.read())
});

// BOUNDED MEMORY (critical for WASM)
const MAX_HISTORY: usize = 1000;
if history.len() >= MAX_HISTORY {
    history.remove(0);
}
history.push(new_value);

// ZERO ALLOCATION (hot paths)
// Instead of Matrix allocations:
pub fn step_2d(&mut self, pos: (f64, f64), grad: (f64, f64)) -> (f64, f64) {
    // Pure scalar math, no heap allocations
    (new_x, new_y)
}
```

---

## Tabs Pattern

```rust
#[derive(Clone, Copy, PartialEq)]
enum Tab { Coefficients, Importance, Correlations }

let mut active_tab = use_signal(|| Tab::Coefficients);

rsx! {
    // Navigation
    div { class: "tabs",
        for tab in [Tab::Coefficients, Tab::Importance, Tab::Correlations] {
            button {
                class: if active_tab() == tab { "tab active" } else { "tab" },
                onclick: move |_| active_tab.set(tab),
                "{tab:?}"
            }
        }
    }

    // Content
    match active_tab() {
        Tab::Coefficients => rsx! { CoefficientDisplay {} },
        Tab::Importance => rsx! { ImportanceChart {} },
        Tab::Correlations => rsx! { HeatmapView {} },
    }
}
```

---

## Progress Indicators

```rust
let mut progress = use_signal(|| 0);

spawn(async move {
    for i in 0..10 {
        process_chunk(i);
        progress.set((i + 1) * 10);
        TimeoutFuture::new(0).await; // Yield to browser
    }
});

rsx! {
    div {
        class: "progress-bar",
        style: "width: {progress()}%",
    }
    p { "Progress: {progress()}%" }
}
```

---

## Accessibility

```rust
rsx! {
    button {
        "aria-label": "Run K-Means algorithm",
        "aria-busy": "{is_processing()}",
        disabled: is_processing(),
        onclick: /* ... */,

        if is_processing() {
            span { "aria-hidden": "true", class: "spinner" }
            span { class: "sr-only", "Processing..." }
        } else {
            "Run"
        }
    }

    div {
        role: "status",
        "aria-live": "polite",
        "{result_message}"
    }
}
```

```css
/* Screen reader only */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0,0,0,0);
}

/* Keyboard focus */
:focus-visible {
  outline: 3px solid var(--primary-color);
  outline-offset: 2px;
}
```

---

## Dark Mode

```rust
// In App component
let mut dark_mode = use_signal(|| false);

use_effect(move || {
    if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
        if let Some(html) = doc.document_element() {
            if dark_mode() {
                html.set_attribute("data-theme", "dark").ok();
            } else {
                html.remove_attribute("data-theme").ok();
            }
        }
    }
});

rsx! {
    button {
        onclick: move |_| dark_mode.set(!dark_mode()),
        if dark_mode() { "☀️" } else { "🌙" }
    }
}
```

---

## Common Mistakes to Avoid

```rust
// ❌ DON'T: Use .unwrap() in WASM (crashes entire app)
let value = signal.read().unwrap();

// ✅ DO: Handle errors gracefully
let value = signal.read().unwrap_or_default();
// Or use panic::catch_unwind for recovery

// ❌ DON'T: Compute in render loop
rsx! {
    for i in 0..1000 {
        div { "{expensive_calc(i)}" }
    }
}

// ✅ DO: Pre-compute
let results = (0..1000).map(expensive_calc).collect::<Vec<_>>();
rsx! { for val in results { div { "{val}" } } }

// ❌ DON'T: Clone unnecessarily
let data = *signal.read().clone();

// ✅ DO: Use reference or Copy
let data = *signal.read(); // For Copy types

// ❌ DON'T: Store derived state
let mut sum = use_signal(|| 0.0);
let mut count = use_signal(|| 0);
let mut average = use_signal(|| 0.0); // Gets out of sync!

// ✅ DO: Compute on demand
let average = sum() / count() as f64;
```

---

## WASM Safety Checklist

- [ ] All algorithm runs wrapped in `panic::catch_unwind`
- [ ] File uploads have size/row/feature limits
- [ ] Long operations have timeouts (5s max)
- [ ] Circular buffers prevent unbounded memory growth
- [ ] User sees progress for operations >1 second
- [ ] Errors display user-friendly messages
- [ ] No `.unwrap()` calls in hot paths
- [ ] Async tasks properly cancelled on unmount

---

## Resources

- **Full Guide**: `/docs/DIOXUS_ML_PLATFORM_GUIDE.md`
- **Dioxus Docs**: https://dioxuslabs.com/learn/0.6/guide/
- **Your Examples**:
  - `/web/src/components/ml_playground.rs` - State & forms
  - `/web/src/components/optimizer_demo.rs` - Performance patterns
  - `/web/src/components/linear_regression_visualizer.rs` - Tabs & viz

---

**Last Updated:** November 8, 2025
**Dioxus Version:** 0.6.0
