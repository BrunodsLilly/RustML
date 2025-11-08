# Performance Benchmark Guide

This document describes how to measure the actual performance of the Optimizer Visualizer to validate that we've met our targets.

## Performance Targets

| Metric | Target | Method |
|--------|--------|--------|
| Iterations/sec | 1000+ | Browser console timing |
| Frame Rate | 60 FPS | Chrome DevTools Performance |
| Memory Usage | Stable | Chrome Task Manager |
| Load Time | <2 sec | Network tab |

---

## Benchmark 1: Iterations Per Second

### Method 1: Browser Console (Quick)

1. Open the Optimizer Visualizer in browser
2. Open Developer Console (F12)
3. Start the optimization (click Play)
4. After 10 seconds, check the iteration counter in the UI
5. Calculate: `iterations / 10 seconds = iter/sec`

**Expected Result:** >1000 iter/sec per optimizer (4000+ total)

### Method 2: Manual Timing Code

Add this to `optimizer_demo.rs` temporarily:

```rust
// In the training loop
if iteration % 1000 == 0 {
    let elapsed = start_time.elapsed().as_secs_f64();
    let rate = iteration as f64 / elapsed;
    console::log_1(&format!("Iteration {}: {:.0} iter/sec", iteration, rate).into());
}
```

---

## Benchmark 2: Frame Rate

### Using Chrome DevTools

1. Open Chrome DevTools (F12)
2. Go to Performance tab
3. Click Record button
4. Start optimization in the visualizer
5. Let run for 10 seconds
6. Stop recording
7. Look at "FPS" graph at top

**Expected Result:** Consistent 60 FPS with no drops

### Using Browser Frame Rate Extension

1. Install "Show FPS" Chrome extension
2. Navigate to optimizer demo
3. Start optimization
4. Monitor FPS counter overlay

**Expected Result:** 60 FPS ±2

---

## Benchmark 3: Memory Stability

### Chrome Task Manager

1. Open Chrome Task Manager (Shift+Esc)
2. Find your tab in the list
3. Note the Memory Footprint
4. Start optimization
5. Monitor memory every minute for 10 minutes

**Expected Result:**
- Initial: ~50-100 MB
- After 10 min: <150 MB (bounded growth)
- No continuous increase

### DevTools Memory Profiler

1. Open DevTools → Memory tab
2. Take heap snapshot (before starting)
3. Start optimization
4. Wait 5 minutes
5. Take another heap snapshot
6. Compare allocations

**Expected Result:**
- Bounded `OptimizerState` path/loss arrays
- No memory leaks
- Total heap growth <50 MB

---

## Benchmark 4: Allocation Profiling

### Verify Zero Allocations in Hot Path

Add instrumentation:

```rust
// In step_2d() method
#[cfg(test)]
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn step_2d(&mut self, position: (f64, f64), gradient: (f64, f64)) -> (f64, f64) {
    // Should be zero new allocations
    let (x, y) = position;
    let (dx, dy) = gradient;
    // ... rest of implementation
}
```

Use `cargo flamegraph` or `perf` to profile:

```bash
cargo build --release -p neural_network
# Run profiling on benchmark
```

**Expected Result:**
- No heap allocations in `step_2d()`
- No `malloc` calls in hot path
- All operations stack-allocated

---

## Benchmark 5: Comparative Performance

### Before vs After (Estimated)

| Metric | Before (Matrix) | After (step_2d) | Improvement |
|--------|----------------|-----------------|-------------|
| Allocations/sec | 24,000 | 0 | ∞ |
| Iterations/sec | 200-500 | 1000+ | 2-5x |
| CPU per iter | High | Low | 10-50x |

### Measuring Actual Improvement

To compare, temporarily revert to Matrix-based approach:

```rust
// Old code (for comparison only)
let mut weights = Matrix::from_vec(vec![x, y], 1, 2)?;
let gradient = Matrix::from_vec(vec![dx, dy], 1, 2)?;
self.optimizer.update_weights(0, &gradient, &mut weights, &shapes);
```

Run same 10-second benchmark and compare iterations/sec.

---

## Automated Benchmark Suite

### Future: Add to CI/CD

Create `benches/optimizer_2d.rs`:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neural_network::optimizer::Optimizer;

fn bench_step_2d_sgd(c: &mut Criterion) {
    let mut opt = Optimizer::sgd(0.01);

    c.bench_function("step_2d_sgd", |b| {
        b.iter(|| {
            let pos = black_box((1.0, 2.0));
            let grad = black_box((0.1, 0.2));
            opt.step_2d(pos, grad)
        });
    });
}

fn bench_step_2d_adam(c: &mut Criterion) {
    let mut opt = Optimizer::adam(0.001, 0.9, 0.999, 1e-8);

    c.bench_function("step_2d_adam", |b| {
        b.iter(|| {
            let pos = black_box((1.0, 2.0));
            let grad = black_box((0.1, 0.2));
            opt.step_2d(pos, grad)
        });
    });
}

criterion_group!(benches, bench_step_2d_sgd, bench_step_2d_adam);
criterion_main!(benches);
```

Run with:
```bash
cargo bench -p neural_network
```

**Expected Results:**
- SGD: <10 ns per step
- Adam: <50 ns per step (more computation)
- Zero allocations confirmed

---

## Visual Regression Testing

### Heatmap Orientation Check

1. **Generate reference screenshot:**
   - Load Rosenbrock function
   - Take screenshot at (0, 0) starting position
   - Save as `test_rosenbrock_reference.png`

2. **After index fix, compare:**
   - Load same function
   - Take screenshot
   - Visual diff should show NO change

3. **Known features to verify:**
   - Minimum at (1, 1) in correct position
   - Valley orientation correct
   - Colors match expected loss values

### Automated Visual Testing (Future)

Use Playwright or similar:

```javascript
// tests/visual-regression.spec.js
test('heatmap renders correctly', async ({ page }) => {
    await page.goto('http://localhost:8080');
    await page.selectOption('#loss-function', 'Rosenbrock');
    await page.screenshot({ path: 'test-output/rosenbrock.png' });

    // Compare with reference
    const diff = await compareImages(
        'test-output/rosenbrock.png',
        'test-reference/rosenbrock.png'
    );

    expect(diff).toBeLessThan(0.01); // <1% difference allowed
});
```

---

## Performance Checklist

### Pre-Release Validation

Before declaring performance targets met, verify:

- [ ] Iterations/sec ≥ 1000 (measured in browser)
- [ ] Frame rate = 60 FPS sustained (DevTools)
- [ ] Memory stable over 10 min run
- [ ] No allocation in `step_2d()` hot path (profiler)
- [ ] Heatmap visual output unchanged (regression test)
- [ ] Load time <2 seconds (Network tab)
- [ ] Smooth animation, no jank (visual inspection)
- [ ] All 4 optimizers perform equally well

### If Targets Not Met

**Iterations/sec < 1000:**
- Profile `step_2d()` implementation
- Check for unexpected allocations
- Verify `#[inline]` on hot functions
- Consider SIMD optimizations

**FPS < 60:**
- Proceed with P1-3 (SVG → Canvas)
- Profile rendering with Chrome DevTools
- Check for layout thrashing
- Reduce heatmap resolution if needed

**Memory Growth:**
- Verify bounded buffers working
- Check for closure captures
- Profile with heap snapshots
- Look for event listener leaks

---

## Reporting Results

After running benchmarks, update `PROGRESS.md`:

```markdown
## 🎯 Benchmark Results

**Date:** [Date]
**Browser:** Chrome 120
**System:** M1 Mac / Intel i7 / etc.

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Iterations/sec | 1000+ | [XXXX] | ✅/❌ |
| Frame Rate | 60 FPS | [XX] FPS | ✅/❌ |
| Memory (10 min) | <150 MB | [XXX] MB | ✅/❌ |
| Load Time | <2 sec | [X.X] sec | ✅/❌ |
```

---

**Last Updated:** November 7, 2025
**Next:** Run actual browser benchmarks and record results
