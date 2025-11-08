# ML Framework Quick Reference

**Quick access to most important patterns from ML_FRAMEWORK_RESEARCH.md**

---

## Current Project: What's Working (Keep This!)

### Zero-Allocation Optimizer Pattern

```rust
// ✅ VALIDATED BY RESEARCH: 3-7% performance gain from zero-garbage APIs
pub fn step_2d(&mut self, pos: (f64, f64), grad: (f64, f64)) -> (f64, f64) {
    // No heap allocations = 10-50x faster than Matrix version
}
```

### Bounded Memory Pattern

```rust
// ✅ INDUSTRY STANDARD for long-running browser apps
const MAX_PATH_LENGTH: usize = 1000;
const MAX_LOSS_HISTORY: usize = 10000;
```

### Current Stack (Minimal & Effective)

```toml
dioxus = { version = "0.6.0", features = ["router"] }
approx = "0.5"  # Good choice for floating-point tests
```

---

## Critical Performance Rules for WASM

### 1. Minimize JS/WASM Boundary Crossings

```rust
// ❌ BAD: Cross boundary 1000+ times
for pixel in pixels {
    set_pixel_js(pixel);  // Each call crosses boundary
}

// ✅ GOOD: Cross once
render_buffer(&pixels);  // Process entire buffer in WASM
```

### 2. Keep Buffers Alive for ImageData

```rust
// ❌ DANGER: Buffer gets dropped → glitchy rendering
let image_data = ImageData::new_with_u8_clamped_array(Clamped(&temp_buffer))?;

// ✅ SAFE: Store buffer as struct field
struct Renderer {
    buffer: Vec<u8>,  // Must outlive ImageData
}
```

### 3. Use black_box in Benchmarks

```rust
// ✅ Prevents compiler from optimizing away code
b.iter(|| {
    let result = optimizer.step_2d(black_box((1.0, 1.0)), black_box((0.1, 0.1)));
    black_box(result)
});
```

---

## When to Add Dependencies

### Phase 1 (Current): Complete Optimizer Visualizer
**Add:** NOTHING - current approach is optimal

### Phase 2: CNN Implementation
**Add:**
```toml
ndarray = "0.17"
ndarray-conv = "0.3"
photon-rs = "0.3"  # Optional: for image processing demos
```

### Phase 3: Advanced Models
**Add ONE of:**
```toml
# Lightweight inference
candle-core = "0.3"

# OR comprehensive training
burn = { version = "0.13", features = ["wgpu"] }
```

---

## Testing Patterns

### Floating-Point Comparisons (Already Using approx ✅)

```rust
use approx::assert_relative_eq;

assert_relative_eq!(result, expected, epsilon = 1e-6);
```

### Property-Based Testing (Add This)

```toml
[dev-dependencies]
proptest = "1.0"
```

```rust
proptest! {
    #[test]
    fn optimizer_decreases_loss(
        (x, y) in (-10.0..10.0, -10.0..10.0),
        lr in 0.0001..0.1,
    ) {
        let mut opt = Optimizer::sgd(lr);
        let loss_before = compute_loss(x, y);
        let (new_x, new_y) = opt.step_2d((x, y), compute_gradient(x, y));
        let loss_after = compute_loss(new_x, new_y);

        prop_assert!(loss_after <= loss_before + 1e-6);
    }
}
```

---

## Canvas Rendering for 60 FPS

### High-Performance Pattern

```rust
pub struct HeatmapRenderer {
    buffer: Vec<u8>,  // Keep alive!
    width: u32,
    height: u32,
}

impl HeatmapRenderer {
    pub fn render(&mut self, loss_fn: impl Fn(f64, f64) -> f64) {
        // 1. Compute all pixels in WASM (no boundary crossings)
        for row in 0..self.height {
            for col in 0..self.width {
                let x = map_to_coords(col);
                let y = map_to_coords(row);
                let loss = loss_fn(x, y);
                let color = value_to_color(loss);

                let idx = ((row * self.width + col) * 4) as usize;
                self.buffer[idx..idx+4].copy_from_slice(&[color.r, color.g, color.b, 255]);
            }
        }

        // 2. Single boundary crossing to render
        let image_data = ImageData::new_with_u8_clamped_array_and_sh(
            Clamped(&self.buffer), self.width, self.height
        ).unwrap();

        self.context.put_image_data(&image_data, 0.0, 0.0).unwrap();
    }
}
```

---

## Browser Benchmarking

### Measure Actual Performance

```javascript
// Run in browser console
async function benchmarkOptimizer() {
    const wasm = await import('./pkg/web.js');
    await wasm.default();

    const iterations = 10000;
    const start = performance.now();

    for (let i = 0; i < iterations; i++) {
        wasm.step_optimizer();
    }

    const elapsed = performance.now() - start;
    const rate = (iterations / elapsed) * 1000;

    console.log(`${rate.toFixed(0)} iterations/sec`);
    return rate;
}

benchmarkOptimizer();
```

### Measure FPS

```javascript
// Chrome DevTools → Performance tab
// Record 10-second session
// Check FPS graph (should be solid green at 60)
```

---

## MNIST Loading Strategy

### For Browser Demo

```rust
// Option 1: Embed small sample (best for demo)
const MNIST_SAMPLE: &[u8] = include_bytes!("../assets/mnist_100.bin");

// Option 2: Progressive loading with caching
async fn load_or_fetch_mnist() -> Result<Vec<u8>, JsValue> {
    // Check IndexedDB cache first
    if let Some(cached) = check_indexeddb("mnist_cache").await? {
        return Ok(cached);
    }

    // Fetch from CDN
    let data = fetch_mnist_cdn().await?;

    // Cache for next time
    store_indexeddb("mnist_cache", &data).await?;

    Ok(data)
}
```

---

## ndarray Best Practices (When You Add It)

### Use High-Order Methods

```rust
// ✅ FAST: Single traversal, optimized
arr.map_inplace(|x| *x *= 2.0);

// ❌ SLOW: Index-based iteration
for i in 0..arr.nrows() {
    for j in 0..arr.ncols() {
        arr[[i, j]] *= 2.0;
    }
}
```

### Use Zip for Parallel Iteration

```rust
Zip::from(&mut output)
    .and(&input1)
    .and(&input2)
    .for_each(|o, &i1, &i2| {
        *o = i1 + i2;
    });
```

### Convolution (ndarray-conv)

```rust
use ndarray_conv::*;

// Automatically uses FFT for large kernels (>11x11)
let output = input.conv(&kernel, ConvMode::Same, PaddingMode::Zeros)?;
```

---

## Decision Trees

### Should I Migrate SVG → Canvas?

```
Measure FPS in browser
    │
    ├─ FPS ≥ 60? → Keep SVG (simpler)
    │
    └─ FPS < 60? → Migrate to Canvas
```

### Which Deep Learning Framework?

```
Need full training pipeline?
    │
    ├─ Yes → Burn (comprehensive)
    │
    └─ No → Need inference?
            │
            ├─ Yes → Candle (lightweight, proven browser demos)
            │
            └─ No → Keep custom code (smallest bundle)
```

### Which Rendering Approach?

```
What are you rendering?
    │
    ├─ Static charts → plotters-canvas
    │
    ├─ Real-time 2D (60 FPS) → Raw Canvas + ImageData
    │
    └─ 3D interactive → WebGPU + wgpu
```

---

## Common Pitfalls to Avoid

### 1. Large Array Passing

```rust
// ❌ wasm-bindgen copies entire array (slow)
#[wasm_bindgen]
pub fn process(data: Vec<f64>) -> Vec<f64> { ... }

// ✅ Use buffer slicing instead
#[wasm_bindgen]
pub fn process(buffer: &mut [f64], offset: usize, length: usize) { ... }
```

### 2. Property Testing with Unbounded Values

```rust
// ❌ Can generate NaN, infinity, overflow
proptest! {
    #[test]
    fn test(a in any::<f64>()) { ... }
}

// ✅ Use bounded ranges
proptest! {
    #[test]
    fn test(a in -100.0..100.0) { ... }
}
```

### 3. Forgetting bias_box in Benchmarks

```rust
// ❌ Compiler may eliminate entire benchmark
b.iter(|| compute_expensive());

// ✅ Prevent optimization
b.iter(|| black_box(compute_expensive()));
```

---

## Performance Targets (Validated)

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Iterations/sec | 1000+ | JS console benchmark |
| Frame Rate | 60 FPS | Chrome DevTools Performance |
| Memory | Stable (bounded) | Chrome DevTools Memory |
| Allocations | 0 in hot path | Criterion.rs benchmarks |
| WASM Bundle | <500 KB | `ls -lh dist/*.wasm` |

---

## Resources (Most Important)

**Essential Docs:**
- wasm-bindgen Guide: https://rustwasm.github.io/docs/wasm-bindgen/
- Dioxus: https://github.com/DioxusLabs/dioxus
- approx: https://docs.rs/approx/latest/approx/

**When Adding Dependencies:**
- ndarray: https://docs.rs/ndarray/latest/ndarray/
- Burn: https://burn.dev/
- Candle: https://github.com/huggingface/candle

**Performance:**
- Criterion.rs: https://bheisler.github.io/criterion.rs/book/
- The Rust Performance Book: https://nnethercote.github.io/perf-book/

---

**See ML_FRAMEWORK_RESEARCH.md for complete documentation**
