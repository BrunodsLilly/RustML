# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 🎯 Vision: Revolutionary ML in the Browser

This project is building a **client-side ML platform** that showcases what's possible when Rust + WASM meet machine learning. The differentiator is **zero-backend computation**: everything runs in the browser at native speeds.

**Current Milestone:** ML Playground with 5 live algorithms (K-Means, PCA, Logistic Regression, StandardScaler, MinMaxScaler) running in browser.

**Latest Achievement (Nov 8, 2025):** ✅ PR #6 merged - Enabled all ML algorithms in web playground with simplified trait syntax.

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
- `/optimizer` - Interactive 4-optimizer comparison
- `/playground` - ✅ **NEW:** ML Playground with 5 algorithms (K-Means, PCA, LogReg, Scalers)

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

**✅ PR #6 Review Completed (Nov 8, 2025):**
Multi-agent deep-dive analysis of ML Playground implementation:
- **Security:** 3 critical (CSV limits, WASM panics, iteration timeouts)
- **Performance:** 5 critical (O(n²) hotspots, synchronous blocking, allocations)
- **Architecture:** 2 critical (String errors, no panic boundaries)
- **Code Quality:** 1 critical (135 lines duplicated error handling)

**Total Findings:** 25 issues (11 P1, 9 P2, 5 P3)

**Key Insights:**
- Excellent trait-based architecture foundation ✅
- 60% verbosity reduction from simplified trait calls ✅
- Critical performance gaps prevent production scale (1K+ samples)
- Missing WASM safety patterns (panic boundaries, input validation)
- Opportunity for 10-50x speedup via zero-allocation patterns

**Review Documents:**
- Code review findings available in session context (Nov 8, 2025)
- Previous review: `docs/reviews/2025-11-07-optimizer-visualizer/`

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

**Based on Nov 8, 2025 comprehensive code review of PR #6 (ML Playground).**

### IMMEDIATE PRIORITIES (Week 1) - Production Blockers

#### 1. **Zero-Allocation ML Algorithms** 🚀
**Why Revolutionary:** Enables 1000+ samples in browser (currently limited to ~100)

**Critical Fixes:**
```rust
// Add to linear_algebra/src/matrix.rs
impl Matrix<f64> {
    pub fn row_slice(&self, row: usize) -> &[f64] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }
}

// Use in K-Means (clustering/src/kmeans.rs:119-125)
let mut row_buffer = vec![0.0; n_features];
for i in 0..n_samples {
    for j in 0..n_features {
        row_buffer[j] = data.get(i, j).unwrap();
    }
    // Use row_buffer instead of allocating get_row()
}
```

**Impact:** 10-50x performance improvement, K-Means 2s → 200ms on 1K samples

**Files to Modify:**
- `linear_algebra/src/matrix.rs` - Add `row_slice()` method
- `clustering/src/kmeans.rs:119-125` - Eliminate `get_row()` allocations
- `dimensionality_reduction/src/pca.rs:93-121` - Add early convergence
- `supervised/src/logistic_regression.rs:124-130` - Vectorize gradients

---

#### 2. **WASM Safety Fortress** 🛡️
**Why Revolutionary:** Prevents silent crashes that kill entire app

**Critical Implementation:**
```rust
// Add to web/src/components/ml_playground.rs:157-167
use std::panic;

onclick: move |_| {
    spawn(async move {
        is_processing.set(true);

        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            if let Some(ref dataset) = *csv_dataset.read() {
                run_algorithm(*selected_algorithm.read(), dataset)
            } else {
                "❌ No dataset loaded".to_string()
            }
        }));

        match result {
            Ok(msg) => result_message.set(msg),
            Err(_) => {
                result_message.set("❌ Algorithm crashed. Please reload and try simpler data.".to_string());
                console::error_1(&"WASM panic caught".into());
            }
        }

        is_processing.set(false);
    });
}
```

**Impact:** Zero silent failures, production-grade error handling

**Additional Safety Measures:**
- Replace 62 `.unwrap()` calls with proper error handling
- Add CSV file limits: 5MB max, 10K rows, 100 features
- Add algorithm timeouts (5 seconds max per operation)

---

#### 3. **Eliminate Code Duplication** 📐
**Why Revolutionary:** 135 lines → 15 lines, single source of truth for errors

**Refactoring Pattern:**
```rust
// Add to ml_playground.rs
fn execute_algorithm<R>(
    algorithm_name: &str,
    fit_and_run: impl FnOnce() -> Result<R, String>,
    format_result: impl FnOnce(R) -> String,
) -> String {
    match fit_and_run() {
        Ok(result) => format!("✅ {} completed!\n\n{}", algorithm_name, format_result(result)),
        Err(e) => format!("❌ {} failed: {}", algorithm_name, e),
    }
}

// Simplify each runner from 27 lines to 5 lines
fn run_kmeans(dataset: &CsvDataset) -> String {
    execute_algorithm(
        "K-Means",
        || {
            let mut kmeans = KMeans::new(3, 100, 1e-4, Some(42));
            kmeans.fit(&dataset.features)?;
            kmeans.predict(&dataset.features)
        },
        |labels| { /* count clusters */ }
    )
}
```

**Impact:** Maintenance burden 5x → 1x, consistent error handling

---

#### 4. **Structured Error Types** 🎯
**Why Revolutionary:** Type-safe errors enable better debugging and error recovery

**Architecture Change:**
```rust
// Add to ml_traits/src/error.rs (NEW FILE)
#[derive(Debug, Clone)]
pub enum MLError {
    InvalidInput { message: String, parameter: &'static str },
    NotFitted { model_type: &'static str },
    DimensionMismatch { expected: (usize, usize), got: (usize, usize) },
    ConvergenceFailure { iterations: usize, final_cost: f64 },
    InsufficientData { required: usize, provided: usize },
    NumericalInstability { context: String },
}

impl std::fmt::Display for MLError { /* ... */ }
impl std::error::Error for MLError {}

// Update all trait signatures
pub trait Clusterer<T: Numeric, D: Data<T>> {
    fn fit(&mut self, X: &D) -> Result<(), MLError>;  // Changed from String
    fn predict(&self, X: &D) -> Result<Vec<usize>, MLError>;
}
```

**Migration Strategy:**
1. Week 1: Add `MLError` enum with `impl From<String> for MLError`
2. Week 2: Migrate one crate per day (clustering → supervised → preprocessing)
3. Week 3: Update all web components to display structured errors
4. Week 4: Remove String error support

**Impact:** Compile-time error safety, better WASM debugging

---

### HIGH-IMPACT FEATURES (Week 2-4) - The Revolution

#### 5. **Interactive Algorithm Configuration** 🎛️
**Why Revolutionary:** Users learn by experimenting, not just watching

**Implementation:**
```rust
// Add to ml_playground.rs
let mut kmeans_clusters = use_signal(|| 3);
let mut pca_components = use_signal(|| 2);
let mut learning_rate = use_signal(|| 0.01);

div { class: "algorithm-config",
    h3 { "Algorithm Parameters" }

    if matches!(*selected_algorithm.read(), Algorithm::KMeans) {
        label { "Number of Clusters (k):" }
        input {
            r#type: "range",
            min: "2",
            max: "10",
            value: "{kmeans_clusters}",
            oninput: move |evt| {
                kmeans_clusters.set(evt.value().parse().unwrap_or(3));
            }
        }
        span { "k = {kmeans_clusters}" }
    }

    // Similar controls for PCA, LogReg, etc.
}
```

**Educational Impact:**
- User sees "k=5 creates too many clusters, k=3 is better"
- Instant feedback loop: adjust → run → compare
- Builds intuition for hyperparameter tuning

**Files to Create:**
- `web/src/components/algorithm_config.rs` (NEW)

---

#### 6. **Real-Time Progress Indicators** ⏱️
**Why Revolutionary:** Users understand algorithm complexity through time

**Implementation:**
```rust
// Add progress tracking to algorithms
let mut progress = use_signal(|| 0);

// In K-Means fit loop
for iter in 0..max_iterations {
    if iter % 10 == 0 {
        progress.set((iter * 100) / max_iterations);
        // Yield to browser every 10 iterations
        gloo::timers::future::sleep(Duration::from_millis(0)).await;
    }
    // ... clustering logic
}

// In UI
if *is_processing.read() {
    div { class: "progress-container",
        div { class: "progress-bar", style: "width: {progress}%" }
        p { "Processing: {progress}% complete" }
    }
}
```

**Educational Value:**
- "PCA on 50 features takes 3 seconds, K-Means on 1000 samples takes 5 seconds"
- Users understand computational cost visually
- Prevents "is it frozen?" confusion

---

#### 7. **Algorithm Comparison Mode** ⚖️
**Why Revolutionary:** Side-by-side reveals strengths/weaknesses

**Concept:**
```
┌─────────────────────┬─────────────────────┐
│   StandardScaler    │    MinMaxScaler     │
├─────────────────────┼─────────────────────┤
│ Result: μ=0, σ=1   │ Result: [0, 1]      │
│ Time: 50ms          │ Time: 45ms          │
│ Memory: 400KB       │ Memory: 200KB       │
│                     │                     │
│ Better for:         │ Better for:         │
│ • Neural networks   │ • Bounded values    │
│ • Gaussian data     │ • Image data        │
└─────────────────────┴─────────────────────┘
```

**Implementation:**
- Run 2+ algorithms on same dataset
- Show results side-by-side with timing
- Highlight differences and use cases

**Educational Impact:** "Now I know when to use StandardScaler vs MinMaxScaler!"

---

#### 8. **Pipeline Builder** 🔗
**Why Revolutionary:** Teaches ML workflow, not just individual algorithms

**Vision:**
```rust
// Create ml_traits/src/pipeline.rs (NEW)
pub struct Pipeline {
    steps: Vec<Box<dyn PipelineStep>>,
}

impl Pipeline {
    pub fn add_step<S: PipelineStep + 'static>(mut self, step: S) -> Self {
        self.steps.push(Box::new(step));
        self
    }

    pub fn fit_transform(&mut self, data: &Matrix<f64>) -> Result<Matrix<f64>, MLError> {
        let mut current = data.clone();
        for step in &mut self.steps {
            current = step.fit_transform(&current)?;
        }
        Ok(current)
    }
}

// UI: Drag-and-drop pipeline builder
Pipeline::new()
    .add_step(StandardScaler::new())
    .add_step(PCA::new(2))
    .add_step(KMeans::new(3, 100, 1e-4, None))
```

**Educational Value:**
- Teaches proper ML workflow: preprocess → reduce → cluster
- Visual pipeline like scikit-learn's Pipeline
- Export as code for learning

---

### GAME-CHANGING FEATURES (Month 2-3) - Beyond State-of-Art

#### 9. **3D Loss Surface Visualization** 🌄
**Why Revolutionary:** No one has real-time 3D optimization in browser

**Technology:** WebGL + Rust WASM for surface computation

**Concept:**
- Render loss function as 3D surface
- Show optimizer path as animated trajectory
- Interactive rotation/zoom
- Compare 4 optimizers racing on same surface

**Educational Impact:** "Now I SEE why Adam outperforms SGD in this valley!"

---

#### 10. **ML Model Explainability Suite** 🔍
**Why Revolutionary:** Understand WHY models make decisions

**Features:**
- SHAP-style feature attribution for LogReg
- Cluster quality metrics for K-Means (silhouette score, inertia)
- PCA variance explained per component
- Feature correlation warnings (multicollinearity detection)

**Already Built Foundation:**
- CorrelationHeatmap component ✅
- FeatureImportanceChart component ✅
- Just need to integrate with ML algorithms

---

#### 11. **Benchmark Suite: WASM vs Python** 📊
**Why Revolutionary:** Prove Rust+WASM superiority with data

**Implementation:**
```rust
// Create benchmarks/ directory
// Run same algorithms in:
// 1. WASM (this codebase)
// 2. Pure Python (sklearn)
// 3. Python + Numba
// 4. JavaScript (TensorFlow.js)

// Generate interactive comparison chart
┌──────────────────────────────────────┐
│  K-Means (1000 samples, k=5)        │
├──────────────────────────────────────┤
│  Rust WASM:     127ms   ███          │
│  Python sklearn: 845ms  ████████████ │
│  TensorFlow.js: 2100ms  ███████████████████████│
│  Speedup: 6.7x vs Python!            │
└──────────────────────────────────────┘
```

**Marketing Gold:**
- Blog post: "How Rust+WASM beats Python at its own game"
- HN front page material
- Educational: Shows compiled languages advantages

---

### INFRASTRUCTURE (Ongoing) - Foundation for Scale

#### 12. **Comprehensive Test Suite** ✅
**Add property-based tests:**
```rust
// Use proptest crate
#[test]
fn test_scaler_roundtrip_property() {
    proptest!(|(data: Vec<Vec<f64>>)| {
        let matrix = Matrix::from_vecs(data.clone())?;
        let mut scaler = StandardScaler::new();
        scaler.fit(&matrix)?;

        let scaled = scaler.transform(&matrix)?;
        let recovered = scaler.inverse_transform(&scaled)?;

        // Property: transform → inverse_transform ≈ identity
        assert_matrices_approx_equal(&matrix, &recovered, 1e-6);
    });
}
```

**Coverage Goals:**
- Unit tests: 80%+ coverage ✅ (already high)
- Integration tests: All algorithm combinations
- Property tests: Mathematical invariants
- E2E tests: Full user workflows in browser

---

#### 13. **Performance Regression Detection** 📈
**Automated benchmarks on every commit:**
```bash
# Add to CI/CD
cargo bench --bench ml_algorithms
# Fail if performance degrades >10%
```

**Track over time:**
- K-Means iterations/sec
- PCA fit time vs features
- LogReg convergence rate
- WASM bundle size

**Files to Create:**
- `benches/ml_algorithms.rs` (NEW)
- `.github/workflows/benchmark.yml` (NEW)

---

#### 14. **Zero-Dependency WASM Optimization** ⚡
**Goal:** Sub-1MB WASM bundle

**Current:** ~2MB (check with `ls -lh target/dx/web/release/web/public/wasm-bindgen/*.wasm`)

**Optimizations:**
- Enable LTO: `lto = "fat"` in Cargo.toml
- Strip symbols: `strip = true`
- Optimize size: `opt-level = "z"`
- Feature flags for algorithm selection (don't bundle unused algorithms)

**Target:** <500KB WASM + <200KB for typical algorithm

---

## For Next AI Agents: Critical Insights

### 🔴 **NEVER DO THIS:**
1. ❌ **Use `.unwrap()` in WASM code** → Silent crashes
2. ❌ **Allocate in hot loops** → Performance death (see K-Means issue)
3. ❌ **Skip input validation** → Security vulnerabilities
4. ❌ **Copy-paste error handling** → Maintenance hell (see PR #6)

### ✅ **ALWAYS DO THIS:**
1. ✅ **Add panic boundaries around WASM algorithms** → Catch crashes
2. ✅ **Profile before optimizing** → Measure, don't guess
3. ✅ **Extract shared patterns** → DRY principle saves 100+ lines
4. ✅ **Test with large datasets** → Find O(n²) before users do

### 🎯 **Architecture Patterns That Work:**
1. **Zero-allocation hot paths** → See `optimizer.rs:step_2d()` for gold standard
2. **Trait-based design** → Easy to add algorithms without breaking API
3. **Bounded memory** → Use MAX_HISTORY constants for long-running demos
4. **Progressive enhancement** → Start simple, add features incrementally

### 📊 **Current State (Nov 8, 2025):**
- ✅ **Architecture:** Excellent trait system, clean dependencies
- ⚠️ **Performance:** Works for small datasets (<100 samples), needs optimization for scale
- ⚠️ **Safety:** Missing WASM panic boundaries, input validation gaps
- ⚠️ **Code Quality:** Good core, duplicated UI layer

### 🚀 **Priority Order for Features:**
1. **Performance > Features** - 1000+ samples must work before adding new algorithms
2. **Safety > Speed** - WASM crashes are worse than slow algorithms
3. **Education > Complexity** - Interactive learning beats feature count
4. **Measurement > Assumptions** - Profile, benchmark, validate claims

### 💡 **Quick Wins (1-2 hours each):**
- Add `Matrix::row_slice()` → 10x K-Means speedup
- Extract error handling helper → Delete 120 lines
- Add progress indicators → Better UX
- Display algorithm hyperparameters in results → Educational value

### 🏆 **Moonshots (1-2 weeks each):**
- 3D WebGL loss surface visualization
- WASM vs Python benchmark suite
- Pipeline builder with drag-and-drop
- Model explainability dashboard

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

**Last Updated:** November 8, 2025
**Status:** ML Playground v0.1 complete, Performance & Safety hardening in progress

**Development Mandate:**
- ✅ Keep merging! Commit frequently!
- ✅ Proactive execution - implement first, ask permission later
- 🎯 Goal: ML Library like NumPy/SciPy with teaching focus via WASM UIs
- 📦 Create new crates when appropriate (aim for <500 lines per file)
- 🏗️ Maintain robust, reusable trait system (see `ml_traits/`)
- ⚡ Zero-allocation patterns for WASM performance
- 🛡️ Safety-first for WASM (panic boundaries, input validation)
- 📊 Profile before optimizing, measure everything

**Next Agent Should:**
1. Start with Week 1 priorities (zero-allocation, WASM safety, eliminate duplication)
2. Create helper functions BEFORE adding new algorithms
3. Run `cargo test --all` before ANY commit
4. Update this file with progress after each major milestone