# Framework Quick Reference

**Fast lookup for common patterns in Dioxus, Rust traits, and WASM optimization.**

---

## Dioxus 0.6.0 Patterns

### Signals (State Management)

```rust
// Create signal
let mut count = use_signal(|| 0);

// Read (tracks dependency)
let value = count();

// Write (triggers re-render)
count.set(10);

// Modify with closure
count.with_mut(|c| *c += 1);

// Global signal
static GLOBAL: GlobalSignal<i32> = Signal::global(|| 0);
```

### Async Operations

```rust
// Event-driven (button click)
spawn(async move {
    let result = fetch().await;
    count.set(result);
});

// Background task (no return value)
use_future(move || async move {
    loop {
        poll_server().await;
        sleep(Duration::from_secs(5)).await;
    }
});

// Reactive fetch (auto re-runs on signal change)
let data = use_resource(move || async move {
    let query = search_query.read();
    api::fetch(&query).await
});

// Access resource data
match &*data.read_unchecked() {
    Some(result) => rsx! { div { "{result}" } },
    None => rsx! { div { "Loading..." } }
}
```

### Long-Running Computation (WASM)

```rust
// Chunk computation to maintain 60 FPS
let run = move |_| {
    spawn(async move {
        loop {
            // 100 iterations per frame
            for _ in 0..100 {
                state.write().step();
            }
            // Yield (16ms @ 60 FPS)
            sleep(Duration::from_millis(16)).await;
            if !running() { break; }
        }
    });
};
```

---

## Rust Trait Patterns

### Associated Types vs Type Parameters

```rust
// Associated type (one implementation per type)
trait Iterator {
    type Item;  // Only ONE item type
    fn next(&mut self) -> Option<Self::Item>;
}

// Type parameter (multiple implementations)
trait From<T> {  // Many T's possible
    fn from(value: T) -> Self;
}

impl From<&str> for String { /* ... */ }
impl From<Vec<u8>> for String { /* ... */ }
```

### Disambiguation Syntax

```rust
// Fully qualified syntax
<Type as Trait>::method(receiver, args)

// Examples
<Person as Greet>::hello(&person);
<&str as Into<String>>::into("hello");

// Turbofish (specify generic parameters)
vec.collect::<Vec<i32>>()
"42".parse::<i32>()
Matrix::<f64>::zeros(3, 3)

// Combined
<Vec<u8> as From<&str>>::from("hello")
```

### Multiple Trait Bounds

```rust
// Basic (+ operator)
fn process<T: Clone + Debug>(value: T) { }

// Where clause (preferred for complex)
fn process<T, U>(a: T, b: U)
where
    T: Clone + Debug + Default,
    U: Into<T>,
{
    // ...
}

// Bound on associated type
fn process<I>(iter: I)
where
    I: Iterator,
    I::Item: Clone,  // Constraint on associated type
{
    // ...
}
```

### Trait with Supertrait Bounds

```rust
pub trait Numeric:
    Copy                          // Required: copyable
    + Debug                       // Required: debuggable
    + std::ops::Add<Output = Self>  // Required: addable
    + std::ops::Sub<Output = Self>  // Required: subtractable
{
    fn zero() -> Self;
    fn sqrt(self) -> Self;
}

// Implementing type must satisfy ALL bounds
impl Numeric for f64 {
    #[inline]
    fn zero() -> Self { 0.0 }

    #[inline]
    fn sqrt(self) -> Self { f64::sqrt(self) }
}
```

---

## WASM Optimization Patterns

### Zero-Allocation Hot Path

```rust
// BAD: Allocates on every iteration
let mut weights = Matrix::from_vec(vec![x, y], 1, 2)?;
optimizer.update(&mut weights);

// GOOD: Stack-allocated tuples
let (x, y) = position;
position = optimizer.step_2d((x, y), (dx, dy));
```

### Bounded Data Structures

```rust
const MAX_HISTORY: usize = 1000;

fn record(&mut self, point: Point) {
    if self.history.len() >= MAX_HISTORY {
        self.history.remove(0);  // Remove oldest
    }
    self.history.push(point);
}
```

### Minimize JS/WASM Boundary Crossings

```rust
// BAD: 1000 calls
for x in data {
    wasm.process_one(x);
}

// GOOD: 1 call
wasm.process_batch(&data);
```

### Avoid Copying (Use Slices)

```rust
// BAD: Copies JS array to Rust Vec
#[wasm_bindgen]
pub fn process(data: Vec<f64>) -> Vec<f64> { }

// GOOD: View into JS memory (no copy)
#[wasm_bindgen]
pub fn process(data: &[f64]) -> Vec<f64> { }

// BEST: Modify in place
#[wasm_bindgen]
pub fn process(data: &mut [f64]) { }
```

### Inline for WASM Performance

```rust
impl Numeric for f64 {
    #[inline]  // Critical for WASM
    fn zero() -> Self { 0.0 }

    #[inline]  // Eliminates function call overhead
    fn abs(self) -> Self { f64::abs(self) }
}
```

---

## Component Patterns

### ML Optimizer Component Structure

```rust
#[derive(Clone, Debug, PartialEq)]
struct OptimizerState {
    optimizer: Optimizer,
    position: (f64, f64),
    path: Vec<(f64, f64)>,      // Bounded history
    losses: Vec<f64>,            // Bounded history
    iteration: usize,
    converged: bool,
}

impl OptimizerState {
    fn new(optimizer: Optimizer, start: (f64, f64)) -> Self { }

    fn reset(&mut self, start: (f64, f64)) { }

    fn step(&mut self, loss_fn: &LossFunction) {
        // Zero-allocation optimization
        let (x, y) = self.position;
        let (dx, dy) = loss_fn.gradient(x, y);
        self.position = self.optimizer.step_2d((x, y), (dx, dy));

        // Bounded history with sampling
        if self.iteration % 10 == 0 {
            if self.path.len() >= MAX_PATH_LENGTH {
                self.path.remove(0);
            }
            self.path.push(self.position);
        }

        self.iteration += 1;
    }
}
```

### Dioxus Component with Async Training

```rust
fn OptimizerDemo() -> Element {
    let mut optimizers = use_signal(|| vec![
        OptimizerState::new(Optimizer::sgd(0.01), (-1.0, 1.0)),
        OptimizerState::new(Optimizer::adam(0.01, 0.9, 0.999), (-1.0, 1.0)),
    ]);

    let mut is_running = use_signal(|| false);

    let run = move |_| {
        if !is_running() {
            is_running.set(true);
            spawn(async move {
                loop {
                    for _ in 0..ITERATIONS_PER_FRAME {
                        optimizers.write().iter_mut().for_each(|o| o.step(&loss_fn));
                    }
                    sleep(Duration::from_millis(16)).await;
                    if !is_running() { break; }
                }
            });
        }
    };

    rsx! {
        button { onclick: run, "Start" }
        button { onclick: move |_| is_running.set(false), "Stop" }
    }
}
```

---

## Linear Algebra in WASM

### nalgebra Setup (no_std)

```toml
[dependencies]
nalgebra = { version = "0.32", default-features = false, features = ["libm"] }
```

```rust
#![no_std]

use nalgebra::{Matrix3, Vector3};

pub fn example() -> Matrix3<f64> {
    let a = Matrix3::new(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    );

    a * Matrix3::identity()  // Stack allocated, no heap
}
```

### Static vs Dynamic Matrices

```rust
// Static: Size known at compile time (FAST)
let static_mat = Matrix3::<f64>::zeros();  // Stack allocated

// Dynamic: Size known at runtime (slower)
let dynamic_mat = DMatrix::<f64>::zeros(3, 3);  // Heap allocated
```

### SIMD in WASM

```toml
[target.wasm32-unknown-unknown]
rustflags = ["-C", "target-feature=+simd128"]

[dependencies]
nalgebra = { version = "0.32", features = ["simd-stable"] }
```

**When SIMD helps:**
- Matrix size > 64x64
- 2-4x performance improvement
- Automatic for supported operations

**When SIMD doesn't help:**
- Small matrices (<64x64)
- Scalar operations (your 2D optimizer)
- Memory copy overhead dominates

---

## Build Configuration

### Cargo.toml (Release Optimization)

```toml
[profile.release]
lto = true           # Link-time optimization
opt-level = "z"      # Optimize for size
codegen-units = 1    # Better optimization, slower compile
panic = "abort"      # Smaller binary (no unwinding)

[profile.wasm-dev]
inherits = "dev"
opt-level = 1        # Faster dev builds for WASM
```

### WASM Post-Processing

```bash
# Build
dx build --release --platform web

# Optimize WASM binary
wasm-opt -Oz -o dist/app_opt.wasm dist/app.wasm

# Check size
ls -lh dist/*.wasm
```

**Expected sizes:**
- Debug: 5-10 MB
- Release: 2-5 MB
- Release + LTO: 1-3 MB
- Release + LTO + wasm-opt: 500 KB - 2 MB

---

## Debugging WASM

### Chrome DevTools

```javascript
// In browser console

// Check memory usage
performance.memory.usedJSHeapSize / 1024 / 1024 + " MB"

// Benchmark iterations
const start = performance.now();
// Run for 10 seconds
const rate = totalIterations / ((performance.now() - start) / 1000);
console.log(`${rate.toFixed(0)} iter/sec`);
```

### Add Telemetry to Rust

```rust
#[wasm_bindgen]
pub fn get_memory_usage(&self) -> usize {
    self.path.len() * std::mem::size_of::<(f64, f64)>()
        + self.losses.len() * std::mem::size_of::<f64>()
}

#[wasm_bindgen]
pub fn get_stats(&self) -> JsValue {
    serde_wasm_bindgen::to_value(&Stats {
        iterations: self.iteration,
        memory_bytes: self.get_memory_usage(),
        path_points: self.path.len(),
    }).unwrap()
}
```

### Console Logging

```rust
use web_sys::console;

// Simple log
console::log_1(&"Hello from Rust!".into());

// With values
console::log_2(
    &"Iteration:".into(),
    &JsValue::from(self.iteration)
);

// Error
console::error_1(&"Something went wrong!".into());
```

---

## Common Pitfalls

### Dioxus

```rust
// WRONG: spawn on every render (creates leak)
fn App() -> Element {
    spawn(async { /* ... */ });  // Called 60 times/sec!
    rsx! { div { "Hello" } }
}

// RIGHT: spawn in event handler
fn App() -> Element {
    rsx! {
        button {
            onclick: move |_| spawn(async { /* ... */ }),
            "Click me"
        }
    }
}
```

### Trait Bounds

```rust
// WRONG: Bounds on type definition (not recommended)
struct MyType<T: Clone + Debug> {
    value: T,
}

// RIGHT: Bounds on impl blocks
struct MyType<T> {
    value: T,
}

impl<T: Clone + Debug> MyType<T> {
    fn process(&self) { /* ... */ }
}
```

### WASM Memory

```rust
// WRONG: Unbounded growth
fn record(&mut self, point: Point) {
    self.history.push(point);  // Grows forever!
}

// RIGHT: Bounded with circular buffer
fn record(&mut self, point: Point) {
    if self.history.len() >= MAX_HISTORY {
        self.history.remove(0);
    }
    self.history.push(point);
}
```

---

## Performance Checklist

### Before Optimizing

- [ ] Profile first (measure, don't guess)
- [ ] Identify hot path (where is time spent?)
- [ ] Set target (what performance is acceptable?)

### Optimization Priority

1. **Algorithm** - O(n²) → O(n log n) beats all micro-optimizations
2. **Allocations** - Remove heap allocations from hot paths
3. **Boundaries** - Minimize JS/WASM crossings
4. **Data structures** - Use bounded buffers, pre-allocate
5. **Inlining** - Add `#[inline]` to small functions
6. **SIMD** - Only for large matrix operations
7. **Binary size** - LTO + wasm-opt for final build

### After Optimizing

- [ ] Measure again (validate improvement)
- [ ] Profile memory (check for leaks)
- [ ] Test edge cases (long runs, large inputs)
- [ ] Document (explain optimization for future you)

---

## Quick Links

**Dioxus:**
- Docs: https://dioxuslabs.com/learn/0.6/
- Examples: https://github.com/DioxusLabs/dioxus/tree/main/examples

**Rust:**
- Book: https://doc.rust-lang.org/book/
- Traits: https://doc.rust-lang.org/book/ch10-02-traits.html

**WASM:**
- wasm-bindgen: https://rustwasm.github.io/wasm-bindgen/
- Rust WASM book: https://rustwasm.github.io/docs/book/

**nalgebra:**
- Docs: https://www.nalgebra.org/
- API: https://docs.rs/nalgebra/latest/nalgebra/

---

**Last Updated:** 2025-11-08
**See Also:** `FRAMEWORK_DOCUMENTATION_RESEARCH.md` for detailed explanations

---

## Visualization Technologies (Added Nov 8, 2025)

### Canvas vs SVG vs WebGL Decision Tree

```
Need visualization?
├─ Real-time 2D (60 FPS) → Canvas + ImageData (20-50x faster)
├─ 3D graphics → Charming (ECharts) or WebGL
├─ Static charts → Plotters or Dioxus SVG
└─ Simple UI elements → Dioxus SVG
```

### Quick Performance Comparison

| Technology | FPS | Interactivity | Bundle Size | Best For |
|------------|-----|---------------|-------------|----------|
| Canvas + ImageData | 60 | Manual | 20 KB | Real-time 2D |
| Charming (ECharts) | 30 | Built-in | 200 KB | 3D surfaces, dashboards |
| Plotters | 10 | None | 50 KB | Static charts, reports |
| Dioxus SVG | 30 | CSS/JS | 5 KB | Simple UI |
| WebGL | 60 | Manual | 30 KB | Custom 3D |

### Recipe: Canvas Real-Time Heatmap (10,000+ iter/sec)

```rust
use web_sys::{CanvasRenderingContext2d, ImageData};
use wasm_bindgen::Clamped;

fn render_heatmap(
    ctx: &CanvasRenderingContext2d,
    width: u32,
    height: u32,
    loss_fn: impl Fn(f64, f64) -> f64
) -> Result<(), JsValue> {
    let mut data = vec![0u8; (width * height * 4) as usize];

    // Manipulate pixels in WASM (fast)
    for y in 0..height {
        for x in 0..width {
            let x_coord = (x as f64 / width as f64) * 10.0 - 5.0;
            let y_coord = (y as f64 / height as f64) * 10.0 - 5.0;
            let loss = loss_fn(x_coord, y_coord);

            let intensity = ((loss / 100.0).clamp(0.0, 1.0) * 255.0) as u8;
            let idx = ((y * width + x) * 4) as usize;
            data[idx + 0] = intensity;
            data[idx + 1] = 0;
            data[idx + 2] = 255 - intensity;
            data[idx + 3] = 255;
        }
    }

    // Single FFI call to render
    let image_data = ImageData::new_with_u8_clamped_array(Clamped(&data), width)?;
    ctx.put_image_data(&image_data, 0.0, 0.0)?;
    Ok(())
}
```

### Recipe: Charming 3D Loss Surface

```rust
use charming::prelude::*;
use charming::renderer::WasmRenderer;

fn create_3d_loss_surface(loss_fn: impl Fn(f64, f64) -> f64) -> Chart {
    let mut data = vec![];
    for x in -50..50 {
        for y in -50..50 {
            data.push(vec![x as f64 * 0.1, y as f64 * 0.1, 
                          loss_fn(x as f64 * 0.1, y as f64 * 0.1)]);
        }
    }

    Chart::new()
        .title(Title::new().text("Loss Surface"))
        .x_axis3d(Axis3D::new().name("X"))
        .y_axis3d(Axis3D::new().name("Y"))
        .z_axis3d(Axis3D::new().name("Loss"))
        .series(Surface3D::new().data(data))
}

// In Dioxus
use_effect(move || {
    spawn(async move {
        let renderer = WasmRenderer::new("3d-chart", 800, 600);
        renderer.render(&create_3d_loss_surface(rosenbrock)).unwrap();
    });
});
```

### Recipe: RequestAnimationFrame Loop (60 FPS)

```rust
use wasm_bindgen::prelude::*;
use std::rc::Rc;
use std::cell::RefCell;

pub fn start_animation_loop<F>(callback: F)
where F: FnMut(f64) + 'static
{
    let f = Rc::new(RefCell::new(None));
    let g = f.clone();
    let mut last_time = 0.0;

    *g.borrow_mut() = Some(Closure::wrap(Box::new(move |time: f64| {
        callback(time - last_time);
        last_time = time;
        request_animation_frame(f.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut(f64)>));

    request_animation_frame(g.borrow().as_ref().unwrap());
}
```

### Performance Pitfalls

❌ **Avoid: Multiple FFI Calls**
```rust
for point in points {
    ctx.fill_rect(point.x, point.y, 2.0, 2.0);  // Slow!
}
```

✅ **Use: Single ImageData Call**
```rust
let image_data = create_image_data(points);
ctx.put_image_data(&image_data, 0.0, 0.0)?;  // 20-50x faster
```

❌ **Avoid: Allocations in Hot Paths**
```rust
let weights = Matrix::from_vec(vec![x, y], 1, 2)?;  // Heap allocation
```

✅ **Use: Stack Tuples**
```rust
let (new_x, new_y) = optimizer.step_2d((x, y), (dx, dy));  // Zero allocation
```

### BrunoML Recommendations

**Optimizer Demo:** Canvas + ImageData → 1000+ iter/sec ✅
**ML Playground:** Charming 3D → Interactive loss surfaces
**Static Reports:** Plotters → Publication quality SVG/PNG
**UI Elements:** Dioxus SVG → Feature bars, coefficients

### Essential Links

- Canvas Examples: https://rustwasm.github.io/docs/wasm-bindgen/examples/2d-canvas.html
- Charming (ECharts): https://github.com/yuankunzhang/charming
- Plotters: https://plotters-rs.github.io/
- Performance Guide: https://rustwasm.github.io/book/reference/time-profiling.html

**See Full Details:** `FRAMEWORK_DOCUMENTATION_RESEARCH.md`

