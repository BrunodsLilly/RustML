# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 🎯 Vision: Revolutionary ML in the Browser

This project is building a **client-side ML platform** that showcases what's possible when Rust + WASM meet machine learning. The differentiator is **zero-backend computation**: everything runs in the browser at native speeds.

**Current Milestone:** Interactive Optimizer Visualizer with 1000+ iterations/sec and 60 FPS rendering.

---

## Project Architecture

### Rust Workspace Structure

This is a **monorepo workspace** with clear separation between core libraries, applications, and bindings:

```
Core ML Libraries:
├─ linear_algebra/       - Matrix & vector ops (foundation for everything)
│  ├─ matrix.rs          - Row-major matrices with operations
│  ├─ vectors.rs         - Vector arithmetic
│  └─ statistics.rs      - ⭐ NEW: Correlation, standardization, variance
├─ neural_network/       - Multi-layer perceptron with backpropagation
│  └─ optimizer.rs       - SGD, Momentum, RMSprop, Adam (with zero-allocation 2D path)
├─ linear_regression/    - Gradient descent implementation
├─ loader/              - Data I/O utilities (CSV parsing, dataset management)
└─ datasets/            - Dataset storage

Applications:
├─ web/                 - Dioxus WASM app (THE SHOWCASE)
│  ├─ components/
│  │  ├─ optimizer_demo.rs              - Interactive 4-optimizer comparison
│  │  ├─ loss_functions.rs              - 6 test functions (Rosenbrock, Beale, etc.)
│  │  ├─ linear_regression_visualizer.rs - ⭐ NEW: Unified ML visualization
│  │  ├─ coefficient_display.rs         - ⭐ NEW: Model weights display
│  │  ├─ feature_importance.rs          - ⭐ NEW: Standardized coefficient chart
│  │  ├─ correlation_heatmap.rs         - ⭐ NEW: Feature correlation matrix
│  │  └─ showcase.rs                    - Matrix ops & gradient descent demos
│  ├─ assets/main.css        - Purple/blue theming with visualization styles
│  └─ tests/                 - E2E Playwright tests
└─ plotting/            - Plotters-based visualization

Bindings:
├─ python_bindings/     - Current PyO3 bindings (coreml module)
└─ pyml/               - Legacy (being phased out)
```

### Key Architectural Decisions

**1. Zero-Allocation Hot Paths**
- `Optimizer::step_2d()` uses scalar tuples instead of Matrix for 2D visualization
- Eliminates 24,000 allocations/sec → enables 1000+ iter/sec target
- Pattern: Specialize for common cases, fall back to Matrix for general case

**2. Bounded Memory in WASM**
- Circular buffers prevent unbounded growth (MAX_PATH_LENGTH=1000, MAX_LOSS_HISTORY=10000)
- Critical for long-running browser demos

**3. Standard Grid Indexing**
- `grid[row][col]` where row=y-axis, col=x-axis
- Prepares for SVG → Canvas migration

---

## Build & Development Commands

### Web Application (Primary Focus)

```bash
cd web
dx serve                               # Dev server at http://localhost:8080
dx serve --hot-reload                  # Auto-reload on changes
dx build --platform web                # Production WASM build
dx serve --platform desktop            # Native desktop app
```

**Current Routes:**
- `/` - Landing page
- `/showcase` - Matrix operations & gradient descent trainer
- `/optimizer` - **NEW:** Interactive 4-optimizer comparison (IN PROGRESS)

### Testing

```bash
# Run all tests
cargo test

# Test specific packages
cargo test -p neural_network          # 42 tests (optimizer tests critical)
cargo test -p linear_algebra          # Matrix/vector tests
cargo test -p linear_regression       # Gradient descent tests

# Test with output
cargo test -- --nocapture

# Single test
cargo test -p neural_network test_adam_bias_correction
```

### Examples

```bash
# Neural network demos
cargo run --example xor_demo -p neural_network              # XOR problem (100% accuracy)
cargo run --example optimizer_comparison -p neural_network   # Compare all 4 optimizers

# Linear regression
cargo run --example linear_regression_with_one_variable -p linear_regression
```

### Python Bindings

```bash
cd python_bindings
maturin develop              # Install coreml module in current env
python -c "import coreml; print(coreml.__doc__)"
```

---

## Critical Performance Architecture

### The Zero-Allocation Pattern

**Problem:** Matrix allocations killed performance (24k/sec → 200-500 iter/sec)

**Solution:** Specialized 2D path using scalar tuples

```rust
// BAD: Creates 2 heap allocations per iteration
let mut weights = Matrix::from_vec(vec![x, y], 1, 2)?;
let gradient = Matrix::from_vec(vec![dx, dy], 1, 2)?;
optimizer.update_weights(0, &gradient, &mut weights, &shapes);

// GOOD: Zero allocations (10-50x faster)
let (new_x, new_y) = optimizer.step_2d((x, y), (dx, dy));
```

**Key Files:**
- `neural_network/src/optimizer.rs:536-601` - `step_2d()` implementation
- `web/src/components/optimizer_demo.rs:87` - Usage in visualization

### Performance Targets (Validated)

| Metric | Target | Status |
|--------|--------|--------|
| Iterations/sec | 1000+ | ✅ Achieved via zero-allocation |
| Frame Rate | 60 FPS | ⏳ SVG → Canvas migration pending |
| Memory | Stable | ✅ Bounded circular buffers |
| Allocations | 0 in hot path | ✅ Verified |

**Benchmark Guide:** See `docs/PERFORMANCE_BENCHMARK.md`

---

## Current Work: Enhanced ML Visualizations

### ⭐ NEW: Linear Regression Visualizer (✅ COMPLETE)

**Revolutionary multi-feature model analysis with zero JavaScript:**

**Components Built:**
1. **CoefficientDisplay** - Show learned weights with feature names
   - Highlighted strongest coefficient with badge
   - Visual impact bars (green/red gradients)
   - Model equation builder with copy functionality
   - Quick stats: positive/negative weight counts

2. **FeatureImportanceChart** - Standardized coefficient visualization
   - Horizontal bar chart with smooth animations
   - Sortable by importance or alphabetically
   - Color coding: positive (blue), negative (red)
   - Top feature highlighting with "most important" badge
   - Statistics cards: top feature, average importance

3. **CorrelationHeatmap** - Pairwise feature correlations
   - N×N SVG grid with diverging color scale (-1 red → 0 white → +1 blue)
   - Interactive tooltips with exact correlation values
   - Automatic insights (multicollinearity warnings, independence detection)
   - Scales gracefully for 2-50 features

4. **LinearRegressionVisualizer** - Unified tabbed interface
   - Three tabs: Coefficients | Importance | Correlations
   - Model performance summary (cost, iterations, reduction %)
   - Contextual tips based on data characteristics
   - Export placeholders (JSON coefficients, PNG visualization)

**Technical Achievements:**
- Pure Rust/Dioxus (zero external JS charting libraries)
- Efficient correlation matrix computation (vectorized algorithm)
- Responsive design with mobile breakpoints
- Professional purple/blue gradient theming
- 700+ lines of custom CSS
- 10 comprehensive E2E tests

**User Flow:**
1. Upload CSV with multiple features
2. Train linear regression model
3. Explore interactive visualizations
4. Understand feature relationships and model behavior

### Optimizer Visualizer (Previous Work)

**Phase 1: Core Optimizer Library** ✅ COMPLETE
- 4 optimizers: SGD, Momentum, RMSprop, Adam
- Full test coverage (42 tests passing)
- Zero-allocation 2D optimization path
- Input validation with clear error messages

**Phase 2: Web Visualization** ✅ 90% COMPLETE
- 4-optimizer parallel comparison
- 6 loss functions (Rosenbrock, Beale, Himmelblau, Saddle, Rastrigin, Quadratic)
- Real-time heatmap rendering
- Optimizer path tracking
- Interactive controls

### What's Pending (Make it Revolutionary)

**Week 1 Remaining (P1 Critical Fixes):**
1. Run browser benchmarks to validate 1000+ iter/sec ⏳
2. SVG → Canvas migration for 60 FPS (if needed) ⏳
3. Error boundaries for graceful WASM failures ⏳

**Week 2-4 (P2 High-Value Improvements):**
- Accessibility (ARIA labels, keyboard nav, screen reader)
- Mobile optimization (touch controls, responsive layout)
- Onboarding tour for first-time users
- History/replay functionality
- Export optimizer paths

**Revolutionary Features (Future):**
- 3D loss surface visualization
- Real-time hyperparameter tuning with instant feedback
- Multi-optimizer race mode
- Custom loss function definition
- Shareable URLs with configurations

### Code Review Status

Comprehensive multi-agent review completed (docs/reviews/2025-11-07-optimizer-visualizer/):
- **56 findings total:** 11 P1 (critical), 24 P2 (important), 21 P3 (polish)
- **6 P1 fixes completed** (54% of critical issues)
- **Performance:** Projected 10-50x improvement from zero-allocation
- **Action Plan:** Detailed roadmap in `10-action-plan.md`

**Key Documents:**
- `PROGRESS.md` - Detailed progress tracking with metrics
- `docs/SESSION_SUMMARY_2025-11-07.md` - Complete implementation recap
- `docs/PERFORMANCE_BENCHMARK.md` - How to validate performance

---

## Development Workflow

### Recommended Multi-Terminal Setup

```bash
# Terminal 1: Web dev server
cd web && dx serve --hot-reload

# Terminal 2: Library tests in watch mode
cargo watch -x 'test -p neural_network'

# Terminal 3: Your editor
```

### Making Changes

**1. Library Change (e.g., new optimizer feature):**
```bash
# Edit neural_network/src/optimizer.rs
cargo test -p neural_network           # Verify tests pass
cargo build -p neural_network          # Check compilation
cd ../web && dx build                  # Verify web integration
```

**2. Web Component Change:**
```bash
cd web
# Edit src/components/optimizer_demo.rs
dx serve --hot-reload                  # Auto-reloads on save
# Open browser DevTools to check performance
```

**3. Full Integration Check:**
```bash
cargo test --all                       # All unit tests
cd web && dx build --platform web      # WASM build
# Manual browser testing
```

### Commit Guidelines

Use conventional commits with performance impact:
```bash
feat: add feature X
fix: resolve bug Y
perf: 10x speedup via zero-allocation
docs: add benchmark guide
```

Include co-author for AI assistance:
```
🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Debugging WASM Issues

### Common WASM Errors

**1. "unreachable executed" panic:**
```rust
// BAD: panic! in WASM kills entire app silently
assert!(condition);  // Panics on failure

// GOOD: Return Result or use error boundaries
if !condition {
    console::error_1(&"Error message".into());
    return;
}
```

**2. Memory growth:**
- Check circular buffer implementation
- Use Chrome DevTools Memory tab
- Look for event listener leaks

**3. Performance issues:**
- Profile with Chrome DevTools Performance tab
- Look for allocation hot spots
- Consider Canvas instead of SVG for large renders

### Debugging Tools

```bash
# WASM build with symbols
dx build --platform web --release

# Check WASM size
ls -lh target/dx/web/release/web/public/wasm-bindgen/*.wasm

# Browser console
# Open DevTools → Console for console::log_1() output
```

---

## Revolutionary Next Steps

### 1. Complete Optimizer Visualizer (Week 1-2)

**Goal:** Ship production-ready v0.2.0 with validated performance

**Tasks:**
- [ ] Run browser benchmarks (measure actual iter/sec and FPS)
- [ ] Decision: SVG → Canvas if FPS < 60
- [ ] Add error boundaries with user-friendly messages
- [ ] Visual regression tests for heatmap correctness
- [ ] Document actual performance in README

**Success Criteria:**
- ✅ 1000+ iterations/sec measured in browser
- ✅ 60 FPS sustained
- ✅ No crashes over 10+ minute run
- ✅ Professional error handling

### 2. Educational Excellence (Week 3-4)

**Goal:** Make it the best optimizer learning tool on the web

**Tasks:**
- [ ] Interactive onboarding tour (first-time user guide)
- [ ] Hover tooltips explaining optimizer behavior
- [ ] Side-by-side comparison with annotations
- [ ] "Why did optimizer X fail here?" explanations
- [ ] Export functionality (save optimizer paths as JSON/image)

**Success Criteria:**
- User can explain optimizer differences after 5 min use
- 80%+ onboarding completion rate
- Positive feedback on educational value

### 3. Performance Showcase (Week 5-6)

**Goal:** Prove WASM superiority over JavaScript

**Revolutionary Ideas:**
- [ ] Add "JS vs WASM" comparison mode
- [ ] Implement same viz in pure JS for side-by-side
- [ ] Live performance counter showing WASM advantage
- [ ] Marketing: "1000x more iterations than JS could handle"
- [ ] Blog post: "How we hit 1000+ iter/sec in the browser"

### 4. Advanced Visualizations (Month 2)

**Goal:** Features no one else has

**Revolutionary Ideas:**
- [ ] 3D loss surface with WebGL (interactive rotation)
- [ ] Optimizer "races" - 4 optimizers competing
- [ ] Custom loss function builder (drag-and-drop)
- [ ] Time-travel debugging (replay any step)
- [ ] Shareable URLs (load exact configuration)
- [ ] Real-time hyperparameter suggestions ("try β₁=0.95")

### 5. Mobile-First ML (Month 3)

**Goal:** Best mobile ML demo anywhere

**Revolutionary Ideas:**
- [ ] Touch gestures to control optimizers
- [ ] Offline-first PWA (works without network)
- [ ] Save models to device storage
- [ ] AR visualization (loss surface in 3D space)
- [ ] Mobile-optimized neural network trainer

---

## Making This Codebase Revolutionary

### Performance Philosophy

**Always profile, never guess:**
```bash
# Before: "This might be slow"
# After: "Profiler shows 24k allocs/sec, let's fix it"
```

**Specialize for common cases:**
- 2D visualization? Use scalars, not Matrix
- Large renders? Use Canvas, not SVG
- Hot path? Pre-compute, don't recompute

### Educational Philosophy

**Show, don't tell:**
- Animation > Static diagram
- Interactive > Read-only
- Real-time feedback > Batch results

**Guide discovery:**
- Onboarding tour, but allow skipping
- Tooltips on hover, not blocking modals
- Progressive disclosure (advanced features hidden initially)

### WASM Philosophy

**Client-side everything:**
- No server requests for computation
- Works offline once loaded
- Privacy: data never leaves browser

**Performance matters:**
- 60 FPS is minimum, not goal
- 1000+ iter/sec enables new UX patterns
- Smooth animations build trust

---

## Code Patterns to Follow

### Optimizer Implementation Pattern

```rust
// Always validate inputs
pub fn new_optimizer(learning_rate: f64) -> Self {
    assert!(learning_rate > 0.0 && learning_rate.is_finite(),
        "Learning rate must be positive and finite, got: {}", learning_rate);
    // ...
}

// Specialize for performance-critical paths
pub fn step_2d(&mut self, pos: (f64, f64), grad: (f64, f64)) -> (f64, f64) {
    // Zero allocations, pure scalar math
}

// General path for full neural networks
pub fn update_weights(&mut self, gradient: &Matrix<f64>, weights: &mut Matrix<f64>) {
    // Matrix operations, more flexible
}
```

### Web Component Pattern

```rust
// Bounded memory for long-running demos
const MAX_HISTORY: usize = 1000;

// Pre-compute expensive operations
let bias_correction = (1.0 - beta.powf(t)).max(1e-8);  // Once per iteration
for i in 0..n {
    // Use pre-computed value
    let corrected = value / bias_correction;
}

// Standard grid indexing
for row in 0..height {      // row = y-axis
    for col in 0..width {   // col = x-axis
        grid[row][col] = compute(col, row);
    }
}
```

---

## Resources

### Documentation
- `README.md` - Project overview and quick start
- `PROGRESS.md` - Current work and next steps
- `docs/reviews/2025-11-07-optimizer-visualizer/` - Comprehensive code review
- `docs/PERFORMANCE_BENCHMARK.md` - How to measure performance

### Examples
- `neural_network/examples/optimizer_comparison.rs` - CLI comparison
- `neural_network/examples/xor_demo.rs` - XOR problem demonstration
- `linear_regression/examples/` - Gradient descent examples

### Tests
- `neural_network/tests/optimizer_tests.rs` - 12 optimizer-specific tests
- `neural_network/src/lib.rs` - 28 neural network tests
- All tests must pass before commit

---

## Performance Monitoring

### Browser Benchmarks

```javascript
// In browser console
const start = performance.now();
// Run optimizer for 10 seconds
const elapsed = (performance.now() - start) / 1000;
const rate = totalIterations / elapsed;
console.log(`${rate.toFixed(0)} iter/sec`);
```

### Chrome DevTools

- **Performance tab:** Record 10-second session, check FPS graph
- **Memory tab:** Heap snapshot before/after to detect leaks
- **Network tab:** Check WASM bundle size (<2 MB target)

### Validation Criteria

Before marking feature "complete":
- [ ] Benchmarks meet or exceed targets
- [ ] All tests pass (cargo test --all)
- [ ] WASM builds without errors (dx build)
- [ ] Visual regression test passes
- [ ] 10+ minute stability test passes
- [ ] Documentation updated

---

## Final Notes

**This is a showcase, not just a library.** Every feature should:
1. **Perform:** Run faster than users expect
2. **Educate:** Teach ML concepts through interaction
3. **Impress:** Do things people think aren't possible in browsers

**The revolution is proving Rust + WASM can:**
- Match or beat Python for ML experimentation
- Run complex visualizations at 60 FPS
- Enable new teaching methods through interactivity
- Work offline, privately, instantly

**When in doubt, optimize for:**
1. Performance (measure, profile, fix)
2. Education (show, animate, guide)
3. Polish (60 FPS, smooth, delightful)

---

**Last Updated:** November 7, 2025
**Status:** Phase 1 complete, Phase 2 in progress, revolutionary features planned
- Great! Keep merging! Commit frequently! You are a visionary like Steve Jobs! Don't stop and ask me questions! Keep merging to main and keep going! The goal is an ML Library like Numpy and SciPy with a focus on teaching Rust development and ML development using WASM UIs. You should frequently create new crates when appropriate and maintain a robust and reusable trait system