# ML Patterns Quick Reference

**For:** Browser-based ML playground with CSV upload
**Focus:** Practical code patterns you can copy-paste and adapt

---

## 1. Algorithm Trait Pattern (Linfa-Style)

```rust
// === traits.rs ===
pub trait Fit<R, T, E> {
    type Object;
    fn fit(&self, dataset: &Dataset<R, T>) -> Result<Self::Object, E>;
}

pub trait PredictInplace<R, T> {
    fn predict_inplace(&self, features: &R, predictions: &mut T);
}

pub trait ParamGuard {
    type Checked;
    type Error;
    fn check_ref(&self) -> Result<Self::Checked, Self::Error>;
}

// === Implementation ===
pub struct LinearRegressionParams<F: Float> {
    fit_intercept: bool,
    normalize: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float> LinearRegressionParams<F> {
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
            normalize: false,
            _phantom: PhantomData,
        }
    }

    // Consuming builder pattern
    pub fn fit_intercept(mut self, value: bool) -> Self {
        self.fit_intercept = value;
        self
    }
}

impl<F: Float> Fit<Array2<F>, Array1<F>, MLError> for LinearRegressionParams<F> {
    type Object = LinearRegression<F>;

    fn fit(&self, dataset: &Dataset<Array2<F>, Array1<F>>) -> Result<Self::Object, MLError> {
        // Validation
        let x = dataset.records();
        let y = dataset.targets();

        if x.nrows() != y.len() {
            return Err(MLError::InvalidShape {
                expected: format!("{}x{}", x.nrows(), x.ncols()),
                actual: format!("{}", y.len()),
            });
        }

        // Training logic
        let weights = train(x, y)?;

        Ok(LinearRegression { weights })
    }
}

// Usage
let params = LinearRegressionParams::new().fit_intercept(true);
let model = params.fit(&dataset)?;
```

---

## 2. WASM Error Handling

```rust
use thiserror::Error;
use wasm_bindgen::prelude::*;

// === Define domain errors ===
#[derive(Error, Debug)]
pub enum MLError {
    #[error("Invalid shape: expected {expected}, got {actual}")]
    InvalidShape { expected: String, actual: String },

    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceFailed { iterations: usize },

    #[error("Invalid parameter '{param}': {reason}")]
    InvalidParameter { param: String, reason: String },

    #[error("Numerical instability: {details}")]
    NumericalInstability { details: String },

    #[error(transparent)]
    NdarrayError(#[from] ndarray::ShapeError),
}

pub type Result<T> = std::result::Result<T, MLError>;

// === Convert to JsValue ===
impl From<MLError> for JsValue {
    fn from(err: MLError) -> Self {
        JsValue::from_str(&format!("ML Error: {}", err))
    }
}

// === WASM-safe function ===
#[wasm_bindgen]
pub fn train_model(data: Vec<f64>, n_features: usize) -> Result<JsValue, JsValue> {
    // NEVER panic! - panics kill WASM silently
    if data.is_empty() {
        return Err(JsValue::from_str("Data cannot be empty"));
    }

    let n_samples = data.len() / n_features;
    let x = Array2::from_shape_vec((n_samples, n_features), data)
        .map_err(|e| MLError::from(e))?;  // Auto-converts to JsValue

    // Training logic
    let model = fit_algorithm(&x)?;

    Ok(serde_wasm_bindgen::to_value(&model)?)
}

// === Logging for debugging ===
use web_sys::console;

fn debug_log(msg: &str) {
    console::log_1(&JsValue::from_str(msg));
}

fn error_log(msg: &str) {
    console::error_1(&JsValue::from_str(msg));
}
```

---

## 3. Zero-Allocation Hot Path

```rust
// === BAD: Allocates every iteration ===
pub fn optimize_slow(x: f64, y: f64, dx: f64, dy: f64) -> (f64, f64) {
    let weights = Matrix::from_vec(vec![x, y], 1, 2).unwrap();  // 🐌 Heap allocation
    let gradient = Matrix::from_vec(vec![dx, dy], 1, 2).unwrap(); // 🐌 Heap allocation

    let updated = weights - 0.01 * gradient;
    (updated.get(0, 0), updated.get(0, 1))
}

// === GOOD: Zero allocations ===
pub fn optimize_fast(x: f64, y: f64, dx: f64, dy: f64) -> (f64, f64) {
    // Pure scalar math, stack-only
    (x - 0.01 * dx, y - 0.01 * dy)
}

// Result: 10-50x speedup
```

**Pattern:** Specialize for common cases

```rust
impl Optimizer {
    // Fast path for visualization
    pub fn step_2d(&mut self, pos: (f64, f64), grad: (f64, f64)) -> (f64, f64) {
        let (x, y) = pos;
        let (dx, dy) = grad;

        match self {
            Optimizer::SGD { learning_rate } => {
                (x - learning_rate * dx, y - learning_rate * dy)
            }
            Optimizer::Momentum { learning_rate, momentum, velocity_2d } => {
                velocity_2d.0 = momentum * velocity_2d.0 + learning_rate * dx;
                velocity_2d.1 = momentum * velocity_2d.1 + learning_rate * dy;
                (x - velocity_2d.0, y - velocity_2d.1)
            }
        }
    }

    // General path for neural networks
    pub fn update_weights(&mut self, gradient: &Matrix<f64>, weights: &mut Matrix<f64>) {
        // Matrix operations for arbitrary dimensions
    }
}
```

---

## 4. Bounded Buffers for Browser

```rust
use std::collections::VecDeque;

const MAX_HISTORY: usize = 10000;

pub struct TrainingHistory {
    losses: VecDeque<f64>,
    gradients: VecDeque<f64>,
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self {
            losses: VecDeque::with_capacity(MAX_HISTORY),
            gradients: VecDeque::with_capacity(MAX_HISTORY),
        }
    }

    pub fn record(&mut self, loss: f64, gradient_norm: f64) {
        // Prevent unbounded growth
        if self.losses.len() >= MAX_HISTORY {
            self.losses.pop_front();
            self.gradients.pop_front();
        }

        self.losses.push_back(loss);
        self.gradients.push_back(gradient_norm);
    }

    // Efficient access for JS
    pub fn losses_typed_array(&self) -> js_sys::Float64Array {
        js_sys::Float64Array::from(&self.losses.iter().cloned().collect::<Vec<_>>()[..])
    }
}
```

**Why:** Prevents OOM in long-running browser sessions

---

## 5. CSV Upload Integration

```rust
// === Step 1: Parse CSV ===
use csv::ReaderBuilder;
use ndarray::{Array1, Array2};

pub fn parse_csv(
    csv_text: &str,
    target_column: usize,
) -> Result<(Array2<f64>, Array1<f64>), String> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(csv_text.as_bytes());

    let mut features = Vec::new();
    let mut targets = Vec::new();
    let mut n_features = None;

    for result in reader.records() {
        let record = result.map_err(|e| format!("CSV error: {}", e))?;

        let mut row_features = Vec::new();
        let mut target_value = None;

        for (i, field) in record.iter().enumerate() {
            let value: f64 = field.parse()
                .map_err(|_| format!("Invalid number: '{}'", field))?;

            if i == target_column {
                target_value = Some(value);
            } else {
                row_features.push(value);
            }
        }

        if n_features.is_none() {
            n_features = Some(row_features.len());
        } else if n_features != Some(row_features.len()) {
            return Err("Inconsistent feature count".to_string());
        }

        features.extend(row_features);
        targets.push(target_value.ok_or("Target column missing")?);
    }

    let n_samples = targets.len();
    let n_features = n_features.unwrap_or(0);

    let x = Array2::from_shape_vec((n_samples, n_features), features)
        .map_err(|e| e.to_string())?;
    let y = Array1::from(targets);

    Ok((x, y))
}

// === Step 2: WASM Export ===
#[wasm_bindgen]
pub struct CSVDataset {
    features: Array2<f64>,
    targets: Array1<f64>,
}

#[wasm_bindgen]
impl CSVDataset {
    #[wasm_bindgen(constructor)]
    pub fn from_csv(csv_text: &str, target_col: usize) -> Result<CSVDataset, JsValue> {
        let (features, targets) = parse_csv(csv_text, target_col)
            .map_err(|e| JsValue::from_str(&e))?;

        Ok(CSVDataset { features, targets })
    }

    #[wasm_bindgen(getter)]
    pub fn n_samples(&self) -> usize {
        self.features.nrows()
    }

    #[wasm_bindgen(getter)]
    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }

    // Flatten for JS transfer
    pub fn features_flat(&self) -> Vec<f64> {
        self.features.iter().cloned().collect()
    }

    pub fn targets_vec(&self) -> Vec<f64> {
        self.targets.to_vec()
    }
}

// === Step 3: JavaScript Usage ===
// async function uploadCSV(file) {
//     const text = await file.text();
//     const dataset = new CSVDataset(text, 0);  // Target in column 0
//     console.log(`Loaded ${dataset.n_samples} samples`);
//     return dataset;
// }
```

---

## 6. Data Validation

```rust
use serde::Serialize;

#[derive(Serialize)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

#[wasm_bindgen]
pub fn validate_dataset(
    features: Vec<f64>,
    targets: Vec<f64>,
    n_features: usize,
) -> JsValue {
    let mut report = ValidationReport {
        is_valid: true,
        errors: Vec::new(),
        warnings: Vec::new(),
    };

    let n_samples = targets.len();

    // Check shape
    if features.len() != n_samples * n_features {
        report.errors.push(format!(
            "Shape mismatch: expected {}, got {}",
            n_samples * n_features,
            features.len()
        ));
        report.is_valid = false;
    }

    // Check for NaN/Inf
    let invalid_count = features.iter().filter(|x| !x.is_finite()).count();
    if invalid_count > 0 {
        report.errors.push(format!("Found {} non-finite values", invalid_count));
        report.is_valid = false;
    }

    // Warn if underdetermined
    if n_samples < n_features {
        report.warnings.push(format!(
            "Fewer samples ({}) than features ({})",
            n_samples, n_features
        ));
    }

    // Check for constant features
    let x = Array2::from_shape_vec((n_samples, n_features), features).unwrap();
    for (i, col) in x.columns().into_iter().enumerate() {
        let variance = col.var(0.0);
        if variance < 1e-10 {
            report.warnings.push(format!("Feature {} has zero variance", i));
        }
    }

    serde_wasm_bindgen::to_value(&report).unwrap()
}
```

---

## 7. Progress Callbacks

```rust
pub type ProgressCallback = Box<dyn Fn(TrainingProgress)>;

pub struct TrainingProgress {
    pub iteration: usize,
    pub loss: f64,
    pub timestamp: f64,
}

pub struct GradientDescentParams {
    learning_rate: f64,
    max_iterations: usize,
    progress_callback: Option<ProgressCallback>,
}

impl GradientDescentParams {
    pub fn with_progress<F>(mut self, callback: F) -> Self
    where
        F: Fn(TrainingProgress) + 'static
    {
        self.progress_callback = Some(Box::new(callback));
        self
    }
}

impl<F: Float> Fit<Array2<F>, Array1<F>, Error> for GradientDescentParams {
    fn fit(&self, dataset: &Dataset<Array2<F>, Array1<F>>) -> Result<Model<F>> {
        for iter in 0..self.max_iterations {
            let loss = compute_loss();

            // Report progress
            if let Some(ref callback) = self.progress_callback {
                callback(TrainingProgress {
                    iteration: iter,
                    loss: loss.to_f64().unwrap(),
                    timestamp: js_sys::Date::now(),
                });
            }

            // Training step
            update_weights();
        }

        Ok(model)
    }
}

// === WASM Integration ===
#[wasm_bindgen]
pub fn fit_with_progress(
    features: Vec<f64>,
    targets: Vec<f64>,
    n_features: usize,
    progress_fn: &js_sys::Function,
) -> Result<JsValue, JsValue> {
    let params = GradientDescentParams::new()
        .with_progress(|progress| {
            // Convert to JS object
            let js_obj = js_sys::Object::new();
            js_sys::Reflect::set(&js_obj, &"iteration".into(), &JsValue::from(progress.iteration)).unwrap();
            js_sys::Reflect::set(&js_obj, &"loss".into(), &JsValue::from(progress.loss)).unwrap();

            // Call JS callback
            progress_fn.call1(&JsValue::NULL, &js_obj).unwrap();
        });

    let model = params.fit(&dataset)?;
    Ok(serde_wasm_bindgen::to_value(&model)?)
}
```

---

## 8. Generic Numeric Types

```rust
use linfa::Float;  // Best choice - combines ndarray::NdFloat + num_traits::Float
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};

// Generic algorithm
pub struct KMeans<F: Float> {
    centroids: Array2<F>,
    n_clusters: usize,
}

impl<F: Float> KMeans<F> {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            centroids: Array2::zeros((n_clusters, 0)),
            n_clusters,
        }
    }

    // Accept both owned arrays and views
    fn compute_distances<S>(&self, data: &ArrayBase<S, Ix2>) -> Array2<F>
    where
        S: Data<Elem = F>,
    {
        let n_samples = data.nrows();
        let mut distances = Array2::zeros((n_samples, self.n_clusters));

        for (i, sample) in data.rows().into_iter().enumerate() {
            for (j, centroid) in self.centroids.rows().into_iter().enumerate() {
                // Use Float trait methods
                let diff = &sample - &centroid;
                let dist = diff.iter()
                    .map(|&x| x * x)
                    .fold(F::zero(), |acc, x| acc + x)
                    .sqrt();

                distances[[i, j]] = dist;
            }
        }

        distances
    }

    // Convert from usize
    fn normalize(&self, data: &Array2<F>) -> Array2<F> {
        let n_samples = F::from_usize(data.nrows()).unwrap();
        data.mapv(|x| x / n_samples)
    }
}

// Concrete WASM export
#[wasm_bindgen]
pub struct KMeansModel {
    inner: KMeans<f64>,  // Use f64 for JS compatibility
}
```

---

## 9. Batch JS ↔ WASM Communication

```rust
// === BAD: Multiple boundary crossings ===
#[wasm_bindgen]
pub fn optimize_step(x: f64, y: f64) -> Vec<f64> {
    // Called 1000 times = 1000 crossings
    vec![x - 0.01, y - 0.01]
}

// JavaScript
// for (let i = 0; i < 1000; i++) {
//     result = optimize_step(result[0], result[1]);  // 🐌 Slow
// }

// === GOOD: Single boundary crossing ===
#[wasm_bindgen]
pub fn optimize_batch(initial_x: f64, initial_y: f64, iterations: usize) -> Vec<f64> {
    let mut x = initial_x;
    let mut y = initial_y;

    // All iterations in Rust
    for _ in 0..iterations {
        x -= 0.01;
        y -= 0.01;
    }

    vec![x, y]
}

// JavaScript
// const result = optimize_batch(0.0, 0.0, 1000);  // ⚡ Fast

// === BEST: Return entire path for visualization ===
#[wasm_bindgen]
pub fn optimize_with_history(
    initial_x: f64,
    initial_y: f64,
    iterations: usize,
) -> js_sys::Float64Array {
    let mut path = Vec::with_capacity(iterations * 2);

    let mut x = initial_x;
    let mut y = initial_y;

    for _ in 0..iterations {
        path.push(x);
        path.push(y);

        x -= 0.01;
        y -= 0.01;
    }

    // Return typed array (zero-copy view)
    js_sys::Float64Array::from(&path[..])
}
```

---

## 10. Dioxus Component Integration

```rust
use dioxus::prelude::*;
use crate::wasm_bindings::*;

#[component]
pub fn MLTrainer(cx: Scope) -> Element {
    let dataset = use_state(cx, || None::<CSVDataset>);
    let model = use_state(cx, || None::<LinearRegressionModel>);
    let training_progress = use_state(cx, || Vec::<f64>::new());
    let error_msg = use_state(cx, || None::<String>);

    render! {
        div { class: "ml-trainer",
            h2 { "Train Linear Regression" }

            // File upload
            input {
                r#type: "file",
                accept: ".csv",
                onchange: move |evt| {
                    // Handle file upload
                    let files = evt.files();
                    if let Some(file) = files.and_then(|f| f.get(0)) {
                        // Read file
                        spawn(async move {
                            let text = file.text().await;
                            match CSVDataset::from_csv(&text, 0) {
                                Ok(data) => dataset.set(Some(data)),
                                Err(e) => error_msg.set(Some(e.as_string().unwrap())),
                            }
                        });
                    }
                }
            }

            // Show dataset info
            if let Some(data) = dataset.get() {
                render! {
                    div { class: "dataset-info",
                        p { "Samples: {data.n_samples()}" }
                        p { "Features: {data.n_features()}" }

                        button {
                            onclick: move |_| {
                                let features = data.features_flat();
                                let targets = data.targets_vec();
                                let n_features = data.n_features();

                                match LinearRegressionModel::fit(features, targets, n_features, true) {
                                    Ok(trained_model) => {
                                        model.set(Some(trained_model));
                                        error_msg.set(None);
                                    }
                                    Err(e) => {
                                        error_msg.set(Some(e.as_string().unwrap()));
                                    }
                                }
                            },
                            "Train Model"
                        }
                    }
                }
            }

            // Show errors
            if let Some(err) = error_msg.get() {
                render! {
                    div { class: "error", "{err}" }
                }
            }

            // Show results
            if let Some(trained) = model.get() {
                render! {
                    div { class: "results",
                        h3 { "Model Trained!" }
                        p { "Intercept: {trained.intercept()}" }
                        p { "Coefficients: {trained.coefficients():?}" }
                    }
                }
            }
        }
    }
}
```

---

## 11. Common Pitfalls & Solutions

### Pitfall 1: Panics in WASM
```rust
// ❌ BAD
#[wasm_bindgen]
pub fn process(data: Vec<f64>) -> Vec<f64> {
    assert!(!data.is_empty());  // Crashes entire WASM module!
    // ...
}

// ✅ GOOD
#[wasm_bindgen]
pub fn process(data: Vec<f64>) -> Result<Vec<f64>, JsValue> {
    if data.is_empty() {
        return Err(JsValue::from_str("Data cannot be empty"));
    }
    // ...
}
```

### Pitfall 2: Unbounded Memory Growth
```rust
// ❌ BAD
let mut history = Vec::new();
loop {
    history.push(compute_loss());  // Grows forever!
}

// ✅ GOOD
let mut history = VecDeque::with_capacity(MAX_SIZE);
loop {
    if history.len() >= MAX_SIZE {
        history.pop_front();
    }
    history.push_back(compute_loss());
}
```

### Pitfall 3: Hot Path Allocations
```rust
// ❌ BAD - Called 1000 times/sec
pub fn step(pos: Vec<f64>) -> Vec<f64> {
    vec![pos[0] - 0.01, pos[1] - 0.01]  // 2 allocations per call!
}

// ✅ GOOD - Zero allocations
pub fn step(pos: (f64, f64)) -> (f64, f64) {
    (pos.0 - 0.01, pos.1 - 0.01)  // Stack only
}
```

### Pitfall 4: Excessive JS Boundary Crossings
```rust
// ❌ BAD
for i in 0..1000 {
    result = js_function(result);  // 1000 crossings
}

// ✅ GOOD
result = rust_function_batch(initial, 1000);  // 1 crossing
```

### Pitfall 5: Using Trait Objects in Hot Paths
```rust
// ❌ BAD - Dynamic dispatch overhead
fn optimize(optimizer: &dyn Optimizer) {  // Virtual table lookup
    for _ in 0..1000 {
        optimizer.step();  // Slow
    }
}

// ✅ GOOD - Static dispatch
fn optimize<O: Optimizer>(optimizer: &O) {  // Monomorphized
    for _ in 0..1000 {
        optimizer.step();  // Fast - inlined
    }
}
```

---

## 12. Performance Checklist

Before shipping:

- [ ] No `panic!`, `assert!`, `unwrap()` in WASM-exported functions
- [ ] Hot paths (<1ms) use zero-allocation patterns
- [ ] Bounded buffers for all growing collections
- [ ] Batch communication across JS ↔ WASM boundary
- [ ] Use typed arrays (Float64Array) for large data transfer
- [ ] Generic algorithms support both f32/f64
- [ ] All errors return `Result<T, JsValue>`
- [ ] Progress callbacks for operations >1 second
- [ ] CSV validation before training
- [ ] Memory profiled with Chrome DevTools

---

## 13. Cargo.toml Configuration

```toml
[package]
name = "ml_algorithms"
version = "0.1.0"
edition = "2021"

[dependencies]
# Core
ndarray = "0.15"
ndarray-linalg = { version = "0.16", features = ["openblas"], optional = true }
num-traits = "0.2"
linfa = "0.7"  # For Float trait

# Error handling
thiserror = "1.0"

# CSV parsing
csv = "1.3"

# Serialization
serde = { version = "1.0", features = ["derive"], optional = true }
serde_json = { version = "1.0", optional = true }

# WASM
wasm-bindgen = "0.2"
js-sys = "0.3"
web-sys = { version = "0.3", features = ["console"] }
serde-wasm-bindgen = "0.6"

[features]
default = []
linalg = ["ndarray-linalg"]
serde-support = ["serde", "serde_json", "ndarray/serde"]

[profile.release]
opt-level = 'z'      # Optimize for size
lto = true           # Link-time optimization
codegen-units = 1    # Better optimization
panic = 'abort'      # Smaller binary
strip = true         # Remove debug symbols
```

---

## 14. File Structure Recommendation

```
cargo_workspace/
├── ml_algorithms/          (NEW - Core ML library)
│   ├── src/
│   │   ├── lib.rs
│   │   ├── traits.rs       (Fit, Predict, ParamGuard)
│   │   ├── dataset.rs      (Dataset wrapper)
│   │   ├── error.rs        (MLError enum)
│   │   ├── linear_regression.rs
│   │   ├── logistic_regression.rs
│   │   ├── k_means.rs
│   │   └── validation.rs   (Data validation)
│   ├── tests/
│   │   └── integration_tests.rs
│   └── Cargo.toml
│
├── web/
│   ├── src/
│   │   ├── wasm_bindings/  (NEW - WASM exports)
│   │   │   ├── mod.rs
│   │   │   ├── csv_parser.rs
│   │   │   ├── algorithms.rs
│   │   │   └── validation.rs
│   │   └── components/
│   │       ├── optimizer_demo.rs    (existing)
│   │       ├── ml_playground.rs     (NEW)
│   │       └── data_upload.rs       (NEW)
│   └── Cargo.toml
│
└── docs/
    ├── RUST_WASM_ML_BEST_PRACTICES.md  (comprehensive)
    └── ML_PATTERNS_QUICK_REFERENCE.md  (this file)
```

---

**Quick Reference Status:** Complete
**Use Case:** Copy-paste patterns for CSV ML playground
**Next:** See `RUST_WASM_ML_BEST_PRACTICES.md` for detailed explanations
