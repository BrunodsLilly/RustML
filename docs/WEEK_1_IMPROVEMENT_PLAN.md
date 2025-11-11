# Week 1 Critical Fixes & Performance Hardening

**Created:** November 8, 2025
**Status:** Post Phase 2 Multi-Agent Review
**Goal:** Fix 4 critical production blockers + eliminate top 11 performance bottlenecks

---

## 📋 Executive Summary

**Review Findings:**
- 41 total issues found (16 P1 Critical, 15 P2 High, 10 P3 Medium)
- Architecture quality: 7.5/10 (excellent foundation, 1 critical bug)
- Performance: 6.0/10 (works <100 samples, needs 10-50x optimization for scale)
- Security: 4.0/10 (136 unwrap() calls, 0 WASM panic boundaries)
- Code quality: 6.5/10 (36-40% duplication, 335 lines YAGNI violations)

**Week 1 Goals:**
1. ✅ Fix critical parameter name mismatch bug (prevents AlgorithmConfigurator from working)
2. 🚀 Add zero-allocation foundation (`Matrix::row_slice()`)
3. 🛡️ Add WASM safety fortress (panic boundaries, input validation)
4. 📐 Eliminate code duplication (135 lines → 15 lines)
5. ⚡ Optimize K-Means hot path (200,000 allocations → 0)

**Success Metrics:**
- 0 critical bugs remaining
- K-Means: 5-10s → <500ms on 1000 samples (10-20x speedup)
- PCA: 10s → <1s on 50 features (10x speedup)
- LogReg: 10s → <2s (5x speedup)
- 0 WASM silent crashes (panic boundaries working)

---

## 🚨 Day 1: Critical Bug Fix (5 minutes)

### Bug #1: Parameter Name Mismatch

**Priority:** P0 - BLOCKING
**Impact:** AlgorithmConfigurator parameter changes don't affect algorithm execution
**Estimated Time:** 5 minutes

**Root Cause:**
- AlgorithmConfigurator sends parameter name "n_clusters" (algorithm_configurator.rs:196)
- MLPlayground checks for "k" (ml_playground.rs:232)
- Name mismatch prevents parameter updates from working

**Files to Change:**
```
web/src/components/ml_playground.rs:232
```

**Fix:**
```rust
// BEFORE (BROKEN):
for param in params.iter() {
    match param.name.as_str() {
        "k" => if let Some(val) = param.current_value.as_i64() {
            current_params.k_clusters = val as usize;
        },
        // ...
    }
}

// AFTER (FIXED):
for param in params.iter() {
    match param.name.as_str() {
        "n_clusters" => if let Some(val) = param.current_value.as_i64() {
            current_params.k_clusters = val as usize;
        },
        // ...
    }
}
```

**Testing:**
1. Run `dx serve --hot-reload`
2. Upload CSV dataset
3. Select K-Means algorithm
4. Change "Number of clusters (k)" slider from 3 to 5
5. Click "Run Algorithm"
6. Verify output shows "Found 5 clusters" (not 3)

**Definition of Done:**
- [ ] Parameter name changed from "k" to "n_clusters"
- [ ] Manual test confirms slider changes affect K-Means output
- [ ] `cargo test -p web` passes
- [ ] Committed with clear message

---

## 🚀 Day 2: Zero-Allocation Foundation (15 minutes)

### Add `Matrix::row_slice()` Method

**Priority:** P1 - CRITICAL FOUNDATION
**Impact:** Enables 10-50x performance improvements across all ML algorithms
**Estimated Time:** 15 minutes

**Problem:**
- Current: `get_row()` allocates a new Vec on every call
- K-Means calls this 200,000 times for 1000 samples × 100 iterations
- PCA, LogReg have similar allocation patterns
- Prevents browser from handling 1000+ sample datasets

**Files to Change:**
```
linear_algebra/src/matrix.rs (add method)
linear_algebra/src/matrix.rs (add tests)
```

**Implementation:**

```rust
// Add to linear_algebra/src/matrix.rs after get_row() method

impl<T: Copy> Matrix<T> {
    /// Get an immutable slice view of a row (zero-copy).
    ///
    /// This is the preferred method for accessing rows in hot paths
    /// as it doesn't allocate. Use `get_row()` only when you need
    /// an owned Vec.
    ///
    /// # Arguments
    /// * `row` - Row index to access
    ///
    /// # Returns
    /// * `Some(&[T])` - Slice view of the row if index is valid
    /// * `None` - If row index is out of bounds
    ///
    /// # Example
    /// ```
    /// let matrix = Matrix::from_vec(vec![1, 2, 3, 4, 5, 6], 2, 3)?;
    /// let row = matrix.row_slice(0).unwrap();
    /// assert_eq!(row, &[1, 2, 3]);
    /// ```
    pub fn row_slice(&self, row: usize) -> Option<&[T]> {
        if row >= self.rows {
            return None;
        }
        let start = row * self.cols;
        let end = start + self.cols;
        Some(&self.data[start..end])
    }
}
```

**Testing:**

```rust
// Add to linear_algebra/src/matrix.rs tests module

#[test]
fn test_row_slice() {
    let matrix = Matrix::from_vec(vec![1, 2, 3, 4, 5, 6], 2, 3).unwrap();

    // Valid row access
    assert_eq!(matrix.row_slice(0), Some(&[1, 2, 3][..]));
    assert_eq!(matrix.row_slice(1), Some(&[4, 5, 6][..]));

    // Out of bounds
    assert_eq!(matrix.row_slice(2), None);
    assert_eq!(matrix.row_slice(100), None);
}

#[test]
fn test_row_slice_zero_copy() {
    let matrix = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();

    // Get slice multiple times - should be instant (no allocations)
    for _ in 0..10000 {
        let _ = matrix.row_slice(0).unwrap();
    }

    // Compare with get_row() which allocates every time
    let slice = matrix.row_slice(0).unwrap();
    let owned = matrix.get_row(0);

    assert_eq!(slice, &owned[..]);
}

#[test]
fn test_row_slice_mutation_safety() {
    let mut matrix = Matrix::from_vec(vec![1, 2, 3, 4], 2, 2).unwrap();

    // row_slice returns immutable reference - can't modify
    let row = matrix.row_slice(0).unwrap();
    assert_eq!(row, &[1, 2]);

    // Can still modify matrix through other means
    matrix.set(0, 0, 10).unwrap();

    // New slice reflects changes
    let row_after = matrix.row_slice(0).unwrap();
    assert_eq!(row_after, &[10, 2]);
}
```

**Validation:**
```bash
cd linear_algebra
cargo test row_slice -- --nocapture
cargo test --all  # Ensure no regressions
```

**Documentation:**
- Add example to `linear_algebra/README.md` showing performance difference
- Document when to use `row_slice()` vs `get_row()`

**Definition of Done:**
- [ ] `row_slice()` method added to Matrix
- [ ] 3 comprehensive tests passing
- [ ] Documentation comments with example
- [ ] `cargo test -p linear_algebra` passes (all 50+ tests)
- [ ] Committed with perf benchmark comment

---

## 🛡️ Day 3-4: WASM Safety Fortress (3 hours)

### Part 1: Panic Boundary Around Algorithms (1.5 hours)

**Priority:** P1 - CRITICAL SAFETY
**Impact:** Prevents silent WASM crashes that kill entire app
**Estimated Time:** 1.5 hours

**Problem:**
- Any `panic!()` or `.unwrap()` failure in WASM causes silent crash
- User sees frozen UI, no error message
- Entire app state lost, requires page reload
- 62 unwrap() calls in web/ crate alone

**Files to Change:**
```
web/src/components/ml_playground.rs:157-167 (Run Algorithm button handler)
```

**Implementation:**

```rust
// Update ml_playground.rs onclick handler

use std::panic;

onclick: move |_| {
    spawn(async move {
        is_processing.set(true);

        // Wrap algorithm execution in panic boundary
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            if let Some(ref dataset) = *csv_dataset.read() {
                let algo = *selected_algorithm.read();
                let params = algorithm_params.read().clone();

                // Run algorithm with current parameters
                match algo {
                    Algorithm::KMeans => run_kmeans(dataset, &params),
                    Algorithm::PCA => run_pca(dataset, &params),
                    Algorithm::LogisticRegression => run_logistic_regression(dataset, &params),
                    Algorithm::StandardScaler => run_standard_scaler(dataset, &params),
                    Algorithm::MinMaxScaler => run_minmax_scaler(dataset, &params),
                }
            } else {
                "❌ No dataset loaded. Please upload a CSV file first.".to_string()
            }
        }));

        match result {
            Ok(message) => {
                result_message.set(message);
            }
            Err(panic_err) => {
                // Log to browser console
                console::error_1(&"Algorithm panic caught!".into());

                // Try to extract panic message
                let panic_msg = if let Some(s) = panic_err.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_err.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };

                console::error_1(&format!("Panic details: {}", panic_msg).into());

                result_message.set(format!(
                    "❌ Algorithm crashed: {}\n\n\
                     This is usually caused by:\n\
                     • Dataset too large (try <100 samples)\n\
                     • Invalid parameter values\n\
                     • Numerical instability\n\n\
                     Please try with a smaller dataset or different parameters.",
                    panic_msg
                ));
            }
        }

        is_processing.set(false);
    });
}
```

**Testing:**

Create manual test cases:
1. **Large dataset crash:** Upload 10,000 row CSV → Should show error, not freeze
2. **Invalid parameters:** Set k=0 for K-Means → Should catch assertion failure
3. **Numerical instability:** All zeros dataset → Should handle gracefully
4. **Memory limit:** Very wide dataset (1000 features) → Should show error

**Validation:**
```bash
cd web
dx serve --hot-reload

# In browser:
# 1. Open DevTools Console
# 2. Upload iris.csv
# 3. Modify code to force panic: panic!("Test panic")
# 4. Verify error message shows, no freeze
# 5. Remove test panic
```

**Definition of Done:**
- [ ] Panic boundary wraps all algorithm execution
- [ ] User-friendly error messages for common failure modes
- [ ] Browser console logs panic details for debugging
- [ ] App remains responsive after panic (no freeze)
- [ ] Manual test of 4 crash scenarios passes

---

### Part 2: Input Validation & Limits (1.5 hours)

**Priority:** P1 - CRITICAL SAFETY
**Impact:** Prevents DoS attacks and resource exhaustion
**Estimated Time:** 1.5 hours

**Problem:**
- No CSV file size limits (can upload 1GB file)
- No row/column limits (can upload 1M rows)
- No bounds checking on algorithm parameters
- Browser tab can hang/crash on oversized data

**Files to Change:**
```
web/src/components/ml_playground.rs:110-140 (CSV upload handler)
web/src/components/shared/algorithm_configurator.rs (parameter validation)
```

**Implementation:**

```rust
// Update CSV upload handler in ml_playground.rs

const MAX_FILE_SIZE: u64 = 5 * 1024 * 1024; // 5MB
const MAX_ROWS: usize = 10_000;
const MAX_FEATURES: usize = 100;

onchange: move |evt: Event<FormData>| {
    spawn(async move {
        upload_status.set("📤 Processing file...".to_string());

        if let Some(file_engine) = &evt.files() {
            let files = file_engine.files();
            if let Some(file_name) = files.get(0) {
                // Validate file size
                if let Ok(file_size) = file_engine.file_size(&file_name) {
                    if file_size > MAX_FILE_SIZE {
                        upload_status.set(format!(
                            "❌ File too large: {:.1}MB\n\
                             Maximum allowed: 5MB\n\n\
                             Try using a smaller dataset or sampling your data.",
                            file_size as f64 / (1024.0 * 1024.0)
                        ));
                        return;
                    }
                }

                if let Ok(contents) = file_engine.read_file(&file_name).await {
                    let text = String::from_utf8_lossy(&contents);

                    // Quick validation: count rows and estimate columns
                    let lines: Vec<&str> = text.lines().collect();
                    let num_rows = lines.len().saturating_sub(1); // Exclude header

                    if num_rows == 0 {
                        upload_status.set("❌ File is empty or has no data rows".to_string());
                        return;
                    }

                    if num_rows > MAX_ROWS {
                        upload_status.set(format!(
                            "❌ Too many rows: {}\n\
                             Maximum allowed: {}\n\n\
                             Try sampling your dataset:\n\
                             • Use first {} rows\n\
                             • Use random sampling\n\
                             • Use stratified sampling",
                            num_rows, MAX_ROWS, MAX_ROWS
                        ));
                        return;
                    }

                    // Estimate columns from header
                    if let Some(header) = lines.get(0) {
                        let num_cols = header.split(',').count();
                        if num_cols > MAX_FEATURES {
                            upload_status.set(format!(
                                "❌ Too many columns: {}\n\
                                 Maximum allowed: {}\n\n\
                                 Try reducing features:\n\
                                 • Remove unnecessary columns\n\
                                 • Use feature selection\n\
                                 • Use PCA for dimensionality reduction",
                                num_cols, MAX_FEATURES
                            ));
                            return;
                        }
                    }

                    // Parse CSV
                    match parse_csv(&text) {
                        Ok(dataset) => {
                            csv_dataset.set(Some(dataset.clone()));
                            upload_status.set(format!(
                                "✅ Loaded: {} rows × {} features\n\
                                 File: {}",
                                dataset.features.rows,
                                dataset.features.cols,
                                file_name
                            ));
                        }
                        Err(e) => {
                            upload_status.set(format!("❌ Parse error: {}", e));
                        }
                    }
                }
            }
        }
    });
}
```

**Add parameter validation to algorithm runners:**

```rust
fn run_kmeans(dataset: &CsvDataset, params: &AlgorithmParams) -> String {
    // Validate parameters
    let k = params.k_clusters;
    let n_samples = dataset.features.rows;

    if k == 0 {
        return "❌ Number of clusters (k) must be at least 1".to_string();
    }

    if k > n_samples {
        return format!(
            "❌ Number of clusters (k={}) cannot exceed number of samples ({})",
            k, n_samples
        );
    }

    if k > 20 {
        return format!(
            "❌ Number of clusters (k={}) is too large (max: 20)\n\
             Large k values:\n\
             • Take longer to compute\n\
             • May not be meaningful\n\
             • Try k=3-10 for most datasets",
            k
        );
    }

    // ... rest of implementation
}
```

**Testing:**

Manual validation tests:
1. Upload 6MB file → Should reject with size error
2. Upload file with 15,000 rows → Should reject with row limit error
3. Upload file with 150 columns → Should reject with column limit error
4. Set k=0 for K-Means → Should show validation error
5. Set k=1000 for 100-sample dataset → Should show validation error

**Definition of Done:**
- [ ] CSV file size limited to 5MB
- [ ] CSV rows limited to 10,000
- [ ] CSV columns limited to 100
- [ ] Algorithm parameters validated before execution
- [ ] Clear error messages guide users to solutions
- [ ] All 5 manual validation tests pass

---

## 📐 Day 5: Eliminate Code Duplication (1 hour)

### Extract Shared Error Handling Helper

**Priority:** P2 - CODE QUALITY
**Impact:** 135 lines → 15 lines, maintenance burden 5x → 1x
**Estimated Time:** 1 hour

**Problem:**
- Each algorithm runner has 27 lines of identical error handling
- 5 algorithms × 27 lines = 135 lines of duplication
- Changes to error format require updating 5 places
- Pattern Recognition agent found 36-40% duplication

**Files to Change:**
```
web/src/components/ml_playground.rs (add helper function at top)
web/src/components/ml_playground.rs:548-684 (simplify all 5 runners)
```

**Implementation:**

```rust
// Add helper function before algorithm runners

/// Execute an ML algorithm with consistent error handling and result formatting.
///
/// This helper eliminates duplication across algorithm runners by providing:
/// - Consistent error message formatting
/// - Success/failure emoji indicators
/// - Standardized result structure
///
/// # Arguments
/// * `algorithm_name` - Display name for the algorithm
/// * `fit_and_run` - Closure that fits the model and generates results
/// * `format_result` - Closure that formats successful results for display
///
/// # Returns
/// Formatted string for UI display with ✅ or ❌ prefix
fn execute_algorithm<R>(
    algorithm_name: &str,
    fit_and_run: impl FnOnce() -> Result<R, String>,
    format_result: impl FnOnce(R) -> String,
) -> String {
    match fit_and_run() {
        Ok(result) => {
            let formatted = format_result(result);
            format!("✅ {} completed successfully!\n\n{}", algorithm_name, formatted)
        }
        Err(error) => {
            format!("❌ {} failed\n\nError: {}\n\nPlease check:\n• Dataset is loaded\n• Parameters are valid\n• Data quality", algorithm_name, error)
        }
    }
}

// Simplify K-Means runner from 27 lines to 12 lines

fn run_kmeans(dataset: &CsvDataset, params: &AlgorithmParams) -> String {
    execute_algorithm(
        "K-Means Clustering",
        || {
            // Validate parameters
            let k = params.k_clusters;
            if k == 0 || k > dataset.features.rows {
                return Err(format!("Invalid k={} for {} samples", k, dataset.features.rows));
            }

            // Fit model
            let mut kmeans = KMeans::new(k, params.kmeans_max_iter, params.kmeans_tolerance, Some(42));
            kmeans.fit(&dataset.features)?;
            kmeans.predict(&dataset.features)
        },
        |labels| {
            // Format results
            let mut cluster_counts = vec![0; params.k_clusters];
            for &label in &labels {
                cluster_counts[label] += 1;
            }

            let mut result = format!("Found {} clusters:\n\n", params.k_clusters);
            for (i, count) in cluster_counts.iter().enumerate() {
                result.push_str(&format!("  Cluster {}: {} samples\n", i, count));
            }
            result
        }
    )
}

// Similarly simplify run_pca, run_logistic_regression, run_standard_scaler, run_minmax_scaler
```

**Metrics:**

Before:
- run_kmeans: 27 lines
- run_pca: 29 lines
- run_logistic_regression: 31 lines
- run_standard_scaler: 25 lines
- run_minmax_scaler: 23 lines
- **Total: 135 lines**

After:
- execute_algorithm helper: 15 lines (shared)
- run_kmeans: 12 lines
- run_pca: 14 lines
- run_logistic_regression: 16 lines
- run_standard_scaler: 10 lines
- run_minmax_scaler: 10 lines
- **Total: 77 lines** (43% reduction)

**Testing:**
```bash
cd web
cargo test -p web  # Ensure no regressions
dx serve --hot-reload

# Manual validation:
# Test each algorithm with success and failure cases
# Verify error messages are consistent
```

**Definition of Done:**
- [ ] `execute_algorithm()` helper function added
- [ ] All 5 algorithm runners refactored to use helper
- [ ] Code reduction: 135 lines → 77 lines (43%)
- [ ] Error message format consistent across algorithms
- [ ] All algorithms still work correctly
- [ ] `cargo test -p web` passes

---

## ⚡ Day 6-7: K-Means Optimization (1.5 hours)

### Replace `get_row()` with `row_slice()` in Hot Path

**Priority:** P1 - PERFORMANCE
**Impact:** 200,000 allocations → 0, estimated 10-20x speedup
**Estimated Time:** 1.5 hours

**Problem:**
- K-Means hot path: distance calculation for every sample to every centroid
- Current: `data.get_row(i)` allocates Vec on every call
- 1000 samples × 3 centroids × 100 iterations = 300,000 allocations
- Performance Oracle estimates 10-50x speedup possible

**Files to Change:**
```
clustering/src/kmeans.rs:119-125 (assign_clusters)
clustering/src/kmeans.rs:152-166 (update_centroids)
```

**Implementation:**

```rust
// BEFORE: clustering/src/kmeans.rs:119-125
fn assign_clusters(&self, data: &Matrix<f64>) -> Vec<usize> {
    let mut labels = vec![0; data.rows];

    for i in 0..data.rows {
        let point = data.get_row(i);  // ALLOCATION!
        let mut min_dist = f64::INFINITY;
        let mut best_cluster = 0;

        for (j, centroid) in self.centroids.iter().enumerate() {
            let dist = euclidean_distance(&point, centroid);  // Owned Vec
            if dist < min_dist {
                min_dist = dist;
                best_cluster = j;
            }
        }
        labels[i] = best_cluster;
    }
    labels
}

// AFTER: Use row_slice() for zero-copy access
fn assign_clusters(&self, data: &Matrix<f64>) -> Vec<usize> {
    let mut labels = vec![0; data.rows];

    for i in 0..data.rows {
        // Zero-copy slice access
        let point = data.row_slice(i).expect("Invalid row index");
        let mut min_dist = f64::INFINITY;
        let mut best_cluster = 0;

        for (j, centroid) in self.centroids.iter().enumerate() {
            // Compute distance directly from slice
            let dist = euclidean_distance_slice(point, centroid);
            if dist < min_dist {
                min_dist = dist;
                best_cluster = j;
            }
        }
        labels[i] = best_cluster;
    }
    labels
}

// Add slice-based distance function
fn euclidean_distance_slice(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "Slices must have same length");

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}
```

**Similarly optimize update_centroids:**

```rust
// BEFORE: clustering/src/kmeans.rs:152-166
fn update_centroids(&mut self, data: &Matrix<f64>, labels: &[usize]) {
    let n_features = data.cols;
    let mut new_centroids = vec![vec![0.0; n_features]; self.k];
    let mut counts = vec![0; self.k];

    for i in 0..data.rows {
        let point = data.get_row(i);  // ALLOCATION!
        let cluster = labels[i];

        for j in 0..n_features {
            new_centroids[cluster][j] += point[j];
        }
        counts[cluster] += 1;
    }

    // ... rest of averaging
}

// AFTER: Direct column access, no row allocations
fn update_centroids(&mut self, data: &Matrix<f64>, labels: &[usize]) {
    let n_features = data.cols;
    let mut new_centroids = vec![vec![0.0; n_features]; self.k];
    let mut counts = vec![0; self.k];

    for i in 0..data.rows {
        let cluster = labels[i];

        // Access each feature directly via get() - no row allocation
        for j in 0..n_features {
            let value = data.get(i, j).expect("Invalid matrix index");
            new_centroids[cluster][j] += value;
        }
        counts[cluster] += 1;
    }

    // Average to get new centroids
    for (cluster, centroid) in new_centroids.iter_mut().enumerate() {
        let count = counts[cluster];
        if count > 0 {
            for value in centroid.iter_mut() {
                *value /= count as f64;
            }
        }
    }

    self.centroids = new_centroids;
}
```

**Performance Testing:**

Create benchmark:
```rust
// clustering/benches/kmeans_bench.rs (NEW FILE)

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use clustering::KMeans;
use linear_algebra::Matrix;

fn bench_kmeans_small(c: &mut Criterion) {
    let data = Matrix::random(100, 4, 0.0, 1.0);

    c.bench_function("kmeans_100_samples", |b| {
        b.iter(|| {
            let mut kmeans = KMeans::new(3, 10, 1e-4, Some(42));
            kmeans.fit(black_box(&data)).unwrap();
        });
    });
}

fn bench_kmeans_medium(c: &mut Criterion) {
    let data = Matrix::random(1000, 10, 0.0, 1.0);

    c.bench_function("kmeans_1000_samples", |b| {
        b.iter(|| {
            let mut kmeans = KMeans::new(3, 100, 1e-4, Some(42));
            kmeans.fit(black_box(&data)).unwrap();
        });
    });
}

criterion_group!(benches, bench_kmeans_small, bench_kmeans_medium);
criterion_main!(benches);
```

Run benchmarks:
```bash
cd clustering
cargo bench --bench kmeans_bench

# Expected results:
# BEFORE: ~5-10 seconds for 1000 samples
# AFTER:  ~300-500ms for 1000 samples (10-20x speedup)
```

**Browser Validation:**
```bash
cd web
dx serve --hot-reload

# Test in browser with timing:
# 1. Upload iris.csv (150 samples)
# 2. Select K-Means, k=3, max_iter=100
# 3. Open DevTools Console
# 4. Run: console.time("kmeans"); <click Run Algorithm>; console.timeEnd("kmeans")
# 5. Should complete in <100ms (was ~500ms before)
```

**Definition of Done:**
- [ ] `assign_clusters()` uses `row_slice()` instead of `get_row()`
- [ ] `update_centroids()` uses direct element access, no row allocations
- [ ] `euclidean_distance_slice()` helper function added
- [ ] Benchmark shows 10-20x speedup on 1000 samples
- [ ] Browser test confirms <100ms for iris dataset
- [ ] All K-Means tests still pass
- [ ] No regressions in other algorithms

---

## 📊 Week 1 Success Criteria

### Quantitative Metrics

**Performance (measured via browser console timing):**
- [ ] K-Means on 1000 samples: <500ms (from 5-10s baseline)
- [ ] K-Means on iris.csv (150 samples): <100ms (from ~500ms)
- [ ] No memory leaks during 10-iteration stress test
- [ ] WASM bundle size <2.5MB (not increased by safety additions)

**Code Quality:**
- [ ] Duplication reduced: 135 lines → <80 lines (40%+ reduction)
- [ ] Test coverage maintained: >80% for modified files
- [ ] No new compiler warnings
- [ ] Documentation updated for all new public APIs

**Safety:**
- [ ] 0 silent WASM crashes in manual testing (5 crash scenarios)
- [ ] All file size limits enforced (tested with oversized files)
- [ ] All parameter validations working (tested with invalid inputs)
- [ ] Panic messages logged to console for debugging

**Bug Fixes:**
- [ ] Parameter name mismatch fixed (AlgorithmConfigurator works)
- [ ] Manual test: slider changes affect algorithm output

### Qualitative Assessment

**User Experience:**
- [ ] Upload large file → Clear error message with guidance
- [ ] Invalid parameters → Helpful validation message
- [ ] Algorithm crashes → App stays responsive, shows error
- [ ] Successful runs → Clear success indicators

**Developer Experience:**
- [ ] Zero-allocation pattern documented with examples
- [ ] Safety patterns documented in CLAUDE.md
- [ ] Performance benchmarks established for regression testing
- [ ] Clear upgrade path for other algorithms (PCA, LogReg)

---

## 📅 Day-by-Day Schedule

| Day | Focus | Time | Tasks | Success Metric |
|-----|-------|------|-------|----------------|
| **Day 1** | Critical Bug | 5 min | Fix parameter name mismatch | Slider works in UI |
| **Day 2** | Foundation | 15 min | Add `Matrix::row_slice()` | Tests pass, docs complete |
| **Day 3** | Safety Part 1 | 1.5 hrs | Add panic boundary | 5 crash tests pass |
| **Day 4** | Safety Part 2 | 1.5 hrs | Input validation & limits | 5 limit tests pass |
| **Day 5** | Code Quality | 1 hr | Extract error handling helper | 40%+ duplication removed |
| **Day 6** | Performance Part 1 | 1 hr | Optimize K-Means assign_clusters | Benchmark 10x faster |
| **Day 7** | Performance Part 2 | 0.5 hrs | Optimize update_centroids, testing | Browser <100ms iris |

**Total Estimated Time:** 6.5 hours (fits in 1 work week with buffer)

---

## 🚀 Post-Week 1: Follow-Up Optimizations

Once Week 1 is complete, apply the same zero-allocation pattern to:

### PCA Optimization (Week 2)
- File: `dimensionality_reduction/src/pca.rs:93-121`
- Problem: Covariance matrix computation allocates rows
- Fix: Use `row_slice()` for dot products
- Impact: 10s → 500ms for 50 features (20x speedup)

### Logistic Regression Optimization (Week 2)
- File: `supervised/src/logistic_regression.rs:124-130`
- Problem: Non-vectorized gradient computation
- Fix: Direct array access via `row_slice()` + SIMD-friendly loop
- Impact: 10s → 1.5s (6.7x speedup)

### Preprocessors (Week 2)
- Files: `preprocessing/src/standard_scaler.rs`, `minmax_scaler.rs`
- Problem: Transform step allocates row Vecs
- Fix: Use `row_slice()` + in-place mutation
- Impact: 2-5x speedup, smoother UI

---

## 📚 References

**Code Review Documents:**
- Phase 2 Multi-Agent Review (Nov 8, 2025) - Session context
- CLAUDE.md sections: "Revolutionary Next Steps", "Current State"

**Key Files:**
- `web/src/components/ml_playground.rs` - Main integration point
- `linear_algebra/src/matrix.rs` - Foundation for all algorithms
- `clustering/src/kmeans.rs` - Performance-critical hot path
- `web/src/components/shared/algorithm_configurator.rs` - Parameter UI

**Performance Baselines:**
- K-Means (1000 samples): 5-10 seconds (before) → <500ms target
- PCA (50 features): 10 seconds (before) → <1s target
- LogReg: 10 seconds (before) → <2s target

---

**Document Status:** Draft v1.0
**Next Review:** After Day 7 completion (update with actual metrics)
**Owner:** Development team
**Created by:** Multi-agent code review system + Claude Code
