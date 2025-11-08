# feat: Enhanced Visualization for Multi-Feature Linear Regression

## Overview

Extend the current CSV upload + linear regression integration to provide rich visualizations for multi-feature models, helping users understand feature importance, correlations, and model behavior.

## Problem Statement

The current linear regression trainer (integrated with CSV upload in PR #3) trains models successfully but provides limited insight into:
- Which features contribute most to predictions
- How features correlate with each other
- Model coefficients for multi-feature datasets
- Prediction quality across different feature combinations

**Impact:** Users cannot easily understand or debug their multi-feature models, limiting the educational value of the platform.

## Proposed Solution

Add interactive visualizations that reveal model internals and data relationships:

1. **Feature Coefficient Display** - Show learned weight for each feature
2. **Feature Importance Ranking** - Visualize relative contribution
3. **Correlation Matrix Heatmap** - Display feature relationships
4. **3D Scatter Plots** - For 2-3 feature datasets (optional)

## Technical Approach

### Architecture

**Component Structure:**
```
web/src/components/
├── linear_regression_visualizer.rs  (NEW - main viz component)
├── coefficient_display.rs           (NEW - coefficients table)
├── correlation_heatmap.rs           (NEW - SVG heatmap)
└── showcase.rs                      (MODIFY - integrate visualizer)
```

**Data Flow:**
```
CsvDataset → LinearRegressor.fit() → TrainedModel
    ↓
TrainedModel → LinearRegressionVisualizer
    ↓
├─→ CoefficientDisplay (weights + bias)
├─→ FeatureImportanceChart (bar chart)
└─→ CorrelationHeatmap (feature correlations)
```

### Implementation Phases

#### Phase 1: Coefficient Display (Week 1, 8-12 hours)

**Tasks:**
- [ ] Create `CoefficientDisplay` component
  - Display weight for each feature with feature name
  - Show bias term separately
  - Highlight largest magnitude coefficients
  - Add copy-to-clipboard for model equation
- [ ] Integrate with `GradientDescentDemo` in showcase.rs
  - Show coefficients after training completes
  - Update when model retrains
- [ ] Add unit tests for coefficient formatting
- [ ] Add E2E test: upload multi-feature CSV, verify coefficients display

**Files to Create:**
```rust
// web/src/components/coefficient_display.rs
#[component]
pub fn CoefficientDisplay(
    weights: Vec<f64>,
    feature_names: Vec<String>,
    bias: f64,
) -> Element {
    // Table with feature names, weights, and relative importance
}
```

**Success Criteria:**
- ✅ Coefficients display within 100ms of training completion
- ✅ All feature names correctly mapped to weights
- ✅ Bias term clearly distinguished
- ✅ Responsive layout for 1-20 features

#### Phase 2: Feature Importance Visualization (Week 2, 12-16 hours)

**Tasks:**
- [ ] Implement importance calculation
  - Normalize weights by feature variance (standardized coefficients)
  - Rank features by absolute importance
  - Handle both positive and negative contributions
- [ ] Create `FeatureImportanceChart` component
  - Horizontal bar chart (SVG)
  - Color coding: positive (blue), negative (red)
  - Sortable by absolute value or alphabetical
  - Interactive tooltips with exact values
- [ ] Add statistical context
  - Show R² score
  - Display feature statistics (min, max, mean, std)
- [ ] Integrate with training flow
- [ ] Add tests for importance calculation
- [ ] Add E2E test: verify importance ranking matches expected order

**Files to Create:**
```rust
// web/src/components/feature_importance.rs
#[component]
pub fn FeatureImportanceChart(
    weights: Vec<f64>,
    feature_names: Vec<String>,
    feature_variances: Vec<f64>,
) -> Element {
    // SVG horizontal bar chart with interactions
}

// Helper functions
fn calculate_standardized_coefficients(weights: &[f64], variances: &[f64]) -> Vec<f64>
fn rank_by_importance(weights: &[f64], names: &[String]) -> Vec<(String, f64)>
```

**Success Criteria:**
- ✅ Importance values correctly normalized
- ✅ Chart scales automatically to data range
- ✅ Hover tooltips show exact contribution
- ✅ Sorting works without re-render flicker

#### Phase 3: Correlation Matrix Heatmap (Week 3, 16-20 hours)

**Tasks:**
- [ ] Implement correlation calculation
  - Pearson correlation coefficient for all feature pairs
  - Handle edge cases (zero variance, NaN)
  - Efficient computation for N×N matrix
- [ ] Create `CorrelationHeatmap` component
  - SVG grid with color-coded cells
  - Diverging color scale: -1 (red) → 0 (white) → +1 (blue)
  - Interactive: hover shows exact correlation value
  - Cell click to highlight corresponding features
  - Legend with interpretation guide
- [ ] Add performance optimizations
  - Memoize correlation matrix (don't recalculate on re-render)
  - Use Canvas for >20 features
  - Bounded rendering for very large datasets
- [ ] Integrate before training
  - Calculate on dataset load
  - Show "Computing correlations..." state
- [ ] Add tests for correlation calculation
- [ ] Add E2E test: verify correlation values, color mapping

**Files to Create:**
```rust
// web/src/components/correlation_heatmap.rs
#[component]
pub fn CorrelationHeatmap(
    correlation_matrix: Vec<Vec<f64>>,
    feature_names: Vec<String>,
) -> Element {
    // SVG heatmap with interactions
}

// linear_algebra/src/correlation.rs (NEW)
pub fn pearson_correlation_matrix(features: &Matrix<f64>) -> Matrix<f64> {
    // Compute correlation between all pairs of columns
}

pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    // Single correlation coefficient
}
```

**Algorithm (Efficient Correlation Calculation):**
```rust
// Vectorized approach using existing Matrix operations
pub fn pearson_correlation_matrix(X: &Matrix<f64>) -> Matrix<f64> {
    let n_features = X.cols;
    let n_samples = X.rows as f64;

    // Center each feature (subtract mean)
    let centered = center_features(X);

    // Compute covariance matrix: (X^T * X) / (n-1)
    let cov = centered.transpose().multiply(&centered).scale(1.0 / (n_samples - 1.0));

    // Normalize by standard deviations to get correlation
    let std_devs = compute_std_devs(&centered);
    normalize_covariance_to_correlation(&cov, &std_devs)
}
```

**Success Criteria:**
- ✅ Correlation matrix computed in <500ms for 20 features, 10k samples
- ✅ Color scale accurately represents -1 to +1 range
- ✅ Symmetric matrix (corr(i,j) == corr(j,i))
- ✅ Diagonal is exactly 1.0 (feature correlates perfectly with itself)
- ✅ Hover interaction shows exact value with 3 decimal precision

#### Phase 4: Integration & Polish (Week 4, 8-12 hours)

**Tasks:**
- [ ] Create unified `LinearRegressionVisualizer` component
  - Tabbed interface: Coefficients | Importance | Correlations
  - Responsive layout for mobile
  - Print-friendly CSS
- [ ] Add export functionality
  - Download coefficients as JSON
  - Export visualizations as PNG/SVG
  - Copy model equation to clipboard
- [ ] Accessibility improvements
  - ARIA labels for all charts
  - Keyboard navigation for tabs
  - Screen reader descriptions
  - High contrast mode support
- [ ] Documentation
  - Add tooltips explaining what each viz shows
  - "How to interpret" guide in help panel
  - Update CLAUDE.md with new components
- [ ] Performance validation
  - Benchmark with 50 features, 100k samples
  - Ensure 60 FPS interactions
  - Memory profiling (bounded growth)
- [ ] Final E2E test suite
  - Complete user workflow: upload → train → explore all visualizations
  - Test edge cases: single feature, 50 features, perfect correlation

**Files to Modify:**
```rust
// web/src/components/showcase.rs
// Add visualizer after training completes (around line 700)
if let Some(trained_model) = trained_model() {
    LinearRegressionVisualizer {
        model: trained_model,
        dataset: csv_dataset(),
    }
}

// web/src/components/mod.rs
pub use linear_regression_visualizer::*;
pub use coefficient_display::*;
pub use feature_importance::*;
pub use correlation_heatmap::*;
```

**Success Criteria:**
- ✅ All visualizations accessible via tabs
- ✅ No visual regression from existing functionality
- ✅ Export works in all major browsers
- ✅ WCAG 2.1 AA compliance
- ✅ All documentation updated

## Alternative Approaches Considered

### 1. Use JavaScript Charting Library (Rejected)
**Pros:** Rich features, proven UX
**Cons:** Breaks WASM-first philosophy, data copying overhead, bundle size

### 2. WebGL for Heatmaps (Future Enhancement)
**Pros:** Much faster for large matrices
**Cons:** Added complexity, not needed for initial launch
**Decision:** Start with SVG, migrate to Canvas/WebGL if needed

### 3. Server-Side Correlation Calculation (Rejected)
**Pros:** Offload computation
**Cons:** Violates "client-side everything" principle, privacy concerns
**Decision:** Keep all computation in WASM

## Acceptance Criteria

### Functional Requirements

- [ ] **Coefficient Display**
  - Shows weight for each feature with name
  - Displays bias term
  - Updates immediately after training
  - Works for 1-50 features

- [ ] **Feature Importance**
  - Calculates standardized coefficients correctly
  - Ranks features by absolute importance
  - Color codes positive/negative contributions
  - Sortable by name or importance

- [ ] **Correlation Heatmap**
  - Computes Pearson correlation for all feature pairs
  - Displays N×N heatmap with diverging color scale
  - Hover shows exact correlation value
  - Handles edge cases (zero variance) gracefully

### Non-Functional Requirements

- [ ] **Performance**
  - Correlation matrix: <500ms for 20 features, 10k samples
  - Rendering: <100ms for all visualizations
  - Interactions: 60 FPS (no jank)
  - Memory: Bounded growth, no leaks

- [ ] **Accessibility**
  - ARIA labels for all charts
  - Keyboard navigation
  - Screen reader support
  - High contrast mode

- [ ] **Cross-Browser**
  - Works in Chrome, Firefox, Safari, Edge
  - Responsive design (mobile + desktop)
  - Print-friendly layouts

### Quality Gates

- [ ] **Test Coverage**
  - Unit tests for correlation calculation (>95% coverage)
  - Unit tests for importance ranking
  - E2E tests for all three visualizations
  - Performance benchmarks in CI

- [ ] **Code Review**
  - Follows project conventions (zero-allocation where applicable)
  - Clear variable names and comments
  - No panics in WASM code
  - Bounded memory usage validated

- [ ] **Documentation**
  - Component API documented with examples
  - User guide for interpreting visualizations
  - CLAUDE.md updated with architecture
  - Performance notes for large datasets

## Success Metrics

**Educational Impact:**
- Users can explain which features matter most (exit survey)
- 80%+ completion rate for "explore visualizations" step
- Positive feedback on "understanding my model"

**Technical Performance:**
- <1 second total visualization time for typical datasets (10 features, 1k samples)
- Zero crashes during 10+ minute sessions
- Memory usage stays <50 MB for all datasets

**Engagement:**
- Average time spent exploring visualizations: >2 minutes
- 50%+ of users try correlation heatmap
- 30%+ export coefficients or charts

## Dependencies & Prerequisites

**Code Dependencies:**
- ✅ CSV upload component (completed in PR #3)
- ✅ Linear regression training (completed in PR #3)
- ✅ Matrix operations library (`linear_algebra` crate)

**New Dependencies:**
- None! All visualization in pure Rust/Dioxus/SVG

**Development Tools:**
- Playwright for E2E tests (already installed)
- Chrome DevTools for performance profiling

## Risk Analysis & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Correlation calculation too slow for large datasets | High | Medium | Use efficient vectorized algorithm, add progress indicator, bound to <50 features |
| SVG heatmap jank with 50×50 grid | Medium | Medium | Switch to Canvas if >20 features, add virtualization |
| Users don't understand correlations | Medium | High | Add interpretation guide, hover tooltips, color legend |
| Memory growth with repeated training | High | Low | Memoize correlations, clear old viz state, bounded history |

## Resource Requirements

**Development Time:** 4 weeks (44-60 hours total)
- Week 1: Coefficient display (8-12 hrs)
- Week 2: Feature importance (12-16 hrs)
- Week 3: Correlation heatmap (16-20 hrs)
- Week 4: Integration & polish (8-12 hrs)

**Team:**
- 1 developer (Rust + Dioxus experience)
- Optional: UX review for color schemes and layouts

**Infrastructure:**
- No additional infrastructure needed
- Runs entirely in browser

## Future Considerations

**Post-Launch Enhancements:**
- 3D scatter plots for 2-3 feature datasets (WebGL)
- Partial dependence plots
- SHAP values for model explainability
- Residual plots and diagnostics
- Interactive feature selection (toggle features on/off)

**Extensibility:**
- Visualizer component reusable for other regression models
- Correlation calculation usable for data exploration
- Export infrastructure usable across all demos

## Documentation Plan

**User-Facing:**
- [ ] Add "Interpreting Visualizations" section to help panel
- [ ] Tooltip for each chart explaining what it shows
- [ ] Example datasets with known patterns (perfect correlation, no correlation)

**Developer-Facing:**
- [ ] Update `CLAUDE.md` with new component architecture
- [ ] API documentation for `LinearRegressionVisualizer`
- [ ] Performance benchmarking guide
- [ ] Add to component showcase

## References & Research

### Internal References

- CSV Upload Integration: `web/src/components/csv_upload.rs`
- Current Training UI: `web/src/components/showcase.rs:488-700`
- Matrix Operations: `linear_algebra/src/lib.rs`
- Performance Patterns: `neural_network/src/optimizer.rs:536-601` (zero-allocation)
- Existing Visualization: `web/src/components/optimizer_demo.rs` (SVG patterns)

### External References

- Pearson Correlation: [Wikipedia](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
- Standardized Coefficients: [Penn State Stats](https://online.stat.psu.edu/stat462/node/132/)
- Correlation Heatmaps: [Seaborn Documentation](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
- Diverging Color Scales: [ColorBrewer](https://colorbrewer2.org/)
- Dioxus Best Practices: [Official Guide](https://dioxuslabs.com/learn/0.6/guide)
- SVG Performance: [MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial)

### Related Work

- PR #3: CSV Data Upload for ML Training
- `optimizer_demo.rs`: SVG visualization patterns
- `loss_functions.rs`: Interactive controls patterns

---

**Labels:** `enhancement`, `visualization`, `linear-regression`, `educational`, `good-first-issue` (Phase 1 only)

**Estimated Effort:** Medium-Large (4 weeks)

**Priority:** P2 (High Value, Not Urgent)
