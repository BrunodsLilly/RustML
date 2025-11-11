# Progress Indicator Design - Issue #18

## Overview

Add real-time progress indicators for long-running ML algorithms to improve UX and prevent "is it frozen?" confusion.

## Current State

**Existing Infrastructure:**
- `is_processing: Signal<bool>` - Binary state (processing / not processing)
- Simple "Processing..." message shown during algorithm execution
- No indication of progress or estimated time remaining
- Users can't tell if algorithm is working or stuck

**User Pain Points:**
- K-Means on 1000 samples: 2-5 seconds with no feedback
- PCA on 50 features: 3-10 seconds of blank screen
- LogisticRegression: Variable time depending on iterations

## Goals

1. **Show iteration progress** - "Iteration 45/100 (45%)"
2. **Real-time updates** - Progress bar fills smoothly
3. **Algorithm-specific details** - Show relevant metrics per algorithm
4. **Non-blocking** - Don't slow down actual computation
5. **Educational** - Help users understand algorithm complexity

## Design

### Component Architecture

```
MLPlayground
├── AlgorithmProgress (NEW)
│   ├── ProgressBar
│   ├── IterationCounter
│   └── MetricsDisplay
├── AlgorithmConfigurator
└── ResultsDisplay
```

### State Management

```rust
// Add to MLPlayground
let mut algorithm_progress = use_signal(|| AlgorithmProgress::default());

#[derive(Clone, Default)]
struct AlgorithmProgress {
    current_iteration: usize,
    total_iterations: usize,
    current_metric: Option<f64>,  // e.g., current loss, inertia
    status_message: String,
}
```

### Algorithm Integration

#### Pattern 1: Callback-Based Progress (K-Means, PCA)

```rust
// In ml_traits/src/clustering.rs
pub trait Clusterer {
    fn fit_with_progress<F>(&mut self, X: &D, progress_callback: F) -> Result<(), String>
    where
        F: FnMut(usize, usize, Option<f64>); // (current, total, metric)
}

// In clustering/src/kmeans.rs
impl KMeans {
    pub fn fit_with_progress<F>(&mut self, X: &Matrix<f64>, mut callback: F) -> Result<(), String>
    where
        F: FnMut(usize, usize, Option<f64>),
    {
        for iteration in 0..self.max_iterations {
            // ... k-means logic ...

            // Report progress every 10 iterations
            if iteration % 10 == 0 {
                callback(iteration, self.max_iterations, Some(inertia));
            }

            // ... convergence check ...
        }
        Ok(())
    }
}
```

#### Pattern 2: Shared Progress State (WASM-friendly)

```rust
// In ml_traits/src/progress.rs (NEW)
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct ProgressReporter {
    state: Arc<Mutex<ProgressState>>,
}

#[derive(Clone, Debug)]
pub struct ProgressState {
    pub current: usize,
    pub total: usize,
    pub metric: Option<f64>,
    pub status: String,
}

impl ProgressReporter {
    pub fn new(total: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(ProgressState {
                current: 0,
                total,
                metric: None,
                status: String::new(),
            })),
        }
    }

    pub fn update(&self, current: usize, metric: Option<f64>, status: &str) {
        if let Ok(mut state) = self.state.lock() {
            state.current = current;
            state.metric = metric;
            state.status = status.to_string();
        }
    }

    pub fn get_state(&self) -> Option<ProgressState> {
        self.state.lock().ok().map(|s| s.clone())
    }
}
```

### UI Component

```rust
// In web/src/components/shared/algorithm_progress.rs (NEW)

#[component]
pub fn AlgorithmProgress(
    current: usize,
    total: usize,
    metric: Option<f64>,
    status: String,
) -> Element {
    let progress_pct = if total > 0 {
        (current as f64 / total as f64 * 100.0).min(100.0)
    } else {
        0.0
    };

    rsx! {
        div { class: "algorithm-progress",
            div { class: "progress-header",
                h3 { "{status}" }
                span { class: "iteration-count",
                    "Iteration {current}/{total}"
                }
            }

            div { class: "progress-bar-container",
                div {
                    class: "progress-bar",
                    style: "width: {progress_pct}%; \
                           background: linear-gradient(90deg, #6c5ce7 0%, #a29bfe {progress_pct}%)",
                }
            }

            div { class: "progress-stats",
                span { class: "progress-percentage",
                    "{progress_pct:.1}% Complete"
                }
                if let Some(metric_value) = metric {
                    span { class: "current-metric",
                        "Current Metric: {metric_value:.4}"
                    }
                }
            }
        }
    }
}
```

### CSS Styling

```css
.algorithm-progress {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.progress-header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: #2d3748;
}

.iteration-count {
    font-family: 'Courier New', monospace;
    font-size: 14px;
    color: #6c5ce7;
    font-weight: 600;
}

.progress-bar-container {
    width: 100%;
    height: 24px;
    background: #f1f3f5;
    border-radius: 12px;
    overflow: hidden;
    margin: 12px 0;
}

.progress-bar {
    height: 100%;
    transition: width 0.3s ease-in-out;
    border-radius: 12px;
}

.progress-stats {
    display: flex;
    justify-content: space-between;
    font-size: 14px;
    color: #718096;
}

.progress-percentage {
    font-weight: 600;
}

.current-metric {
    font-family: 'Courier New', monospace;
    color: #6c5ce7;
}
```

## Implementation Plan

### Phase 1: Infrastructure (1-2 hours)

1. **Create ProgressReporter trait**
   - Location: `ml_traits/src/progress.rs` (NEW)
   - Add to `ml_traits/src/lib.rs`
   - Design: Arc<Mutex<ProgressState>> for thread-safe shared state

2. **Create AlgorithmProgress component**
   - Location: `web/src/components/shared/algorithm_progress.rs` (NEW)
   - Props: current, total, metric, status
   - Responsive progress bar with smooth animations

3. **Add CSS styling**
   - Location: `web/assets/main.css`
   - Purple/blue gradient theme matching existing design

### Phase 2: K-Means Integration (30 min)

1. **Update K-Means algorithm**
   - Add optional ProgressReporter parameter to `fit()`
   - Call `reporter.update()` every 10 iterations
   - Location: `clustering/src/kmeans.rs:243-282`

2. **Update MLPlayground**
   - Create ProgressReporter before calling `run_kmeans()`
   - Poll progress state in async loop
   - Display AlgorithmProgress component when `is_processing`

### Phase 3: PCA Integration (30 min)

1. **Update PCA algorithm**
   - Add progress reporting to power iteration loop
   - Report component extraction progress
   - Location: `dimensionality_reduction/src/pca.rs:158-197`

2. **Update `run_pca()` in MLPlayground**
   - Same pattern as K-Means

### Phase 4: LogisticRegression Integration (30 min)

1. **Update LogisticRegression**
   - Add progress reporting to gradient descent
   - Report current loss value as metric
   - Location: `supervised/src/logistic_regression.rs:100-164`

2. **Update `run_logistic_regression()` in MLPlayground**
   - Show loss reduction progress

### Phase 5: Testing & Polish (1 hour)

1. **Manual testing**
   - Test with various dataset sizes
   - Verify smooth progress updates
   - Check different browsers

2. **E2E tests** (optional for v1)
   - Verify progress indicator appears
   - Check progress reaches 100%

3. **Documentation**
   - Update CLAUDE.md with progress indicator feature
   - Add example screenshots

## Alternative Approaches

### Option 1: Channel-Based Progress (Async-friendly)

```rust
use tokio::sync::mpsc;

let (tx, mut rx) = mpsc::channel(100);

// In algorithm
tx.send(ProgressUpdate { current: i, total, metric }).await;

// In UI
spawn(async move {
    while let Some(update) = rx.recv().await {
        algorithm_progress.set(update);
    }
});
```

**Pros:** Clean separation, async-native
**Cons:** Requires tokio runtime in WASM (complex)

### Option 2: Direct State Mutation (Simplest)

```rust
let progress = use_signal(|| (0, 100));

// In algorithm (passed by reference)
*progress.write() = (current_iter, max_iter);
```

**Pros:** Simplest implementation
**Cons:** Tight coupling between algorithm and UI

**Decision:** Use Arc<Mutex<ProgressState>> (shared state) for v1
- Works in WASM without extra dependencies
- Clean separation of concerns
- Easy to poll from UI async task

## Success Metrics

1. **User sees progress within 100ms** of algorithm start
2. **Progress updates at least every 500ms** during execution
3. **Final progress = 100%** when algorithm completes
4. **No performance regression** - <5% overhead from progress tracking
5. **Works on all browsers** (Chromium, Firefox, WebKit)

## Future Enhancements

1. **Estimated Time Remaining**
   - Track iteration speed
   - Calculate ETA based on average time/iteration

2. **Cancellation Support**
   - Add "Cancel" button to progress indicator
   - Gracefully stop algorithm mid-execution

3. **Progress History**
   - Store progress snapshots
   - Show progress graph after completion

4. **Multi-step Progress**
   - Data loading: 0-20%
   - Training: 20-80%
   - Evaluation: 80-100%

---

**Created:** November 9, 2025
**Status:** Design Complete - Ready for Implementation
**Priority:** P1 (High UX Impact)
**Estimated Time:** 3-4 hours total
