# ML Playground - Critical UX Bugs Report

**Date:** November 9, 2025
**Status:** CRITICAL - Playground is non-functional
**Priority:** P0 - Blocks all user value

---

## Executive Summary

Comprehensive Playwright audit reveals **the ML Playground is completely broken**. Core functionality does not work:
- ❌ Algorithm selection dropdown missing
- ❌ Parameter sliders don't exist or don't work
- ❌ Run button missing
- ❌ Results not displayed
- ❌ No train/test split
- ❌ No predictions visualization
- ❌ No summary statistics

**User Impact:** 100% - No value delivered. Users cannot use any ML algorithms.

---

## Critical Bugs (P0 - Production Blockers)

### BUG #1: Algorithm Selection Dropdown Missing
**Status:** ❌ CONFIRMED
**Severity:** CRITICAL
**Test:** `ml-playground-audit.spec.js:62`

**Issue:**
```
Error: expect(locator).toBeVisible() failed
Locator: locator('select')
Expected: visible
Error: element(s) not found
```

**Impact:** Users cannot select which algorithm to run (K-Means, PCA, LogReg, etc.)

**Fix Required:**
- Add `<select>` dropdown with algorithm options
- Populate with: K-Means, PCA, Logistic Regression, StandardScaler, MinMaxScaler
- Wire up selection to state management

---

### BUG #2: Run Algorithm Button Missing
**Status:** ❌ CONFIRMED
**Severity:** CRITICAL
**Test:** `ml-playground-audit.spec.js:119`

**Issue:**
```
Error: expect(locator).toBeVisible() failed
Locator: locator('button').filter({ hasText: /run/i })
Expected: visible
Error: element(s) not found
```

**Impact:** Users cannot execute algorithms even if they select them

**Fix Required:**
- Add "Run Algorithm" button
- Connect to algorithm execution logic
- Show loading state during execution
- Display results after completion

---

### BUG #3: CSV Upload Input Hidden
**Status:** ❌ CONFIRMED
**Severity:** CRITICAL
**Test:** `ml-playground-audit.spec.js:27`

**Issue:**
```
Error: expect(locator).toBeVisible() failed
Locator: locator('input[type="file"][accept=".csv"]')
Expected: visible
Received: hidden
```

**Impact:** File input exists but is hidden - poor UX

**Current State:**
```rust
input {
    r#type: "file",
    accept: ".csv",
    id: "csv-upload",
    // Missing style to make visible
}
```

**Fix Required:**
- Make file input visible OR
- Keep hidden but ensure label triggers it correctly

---

### BUG #4: Results Not Displayed After Running Algorithm
**Status:** ❌ CONFIRMED
**Severity:** CRITICAL
**Test:** `ml-playground-audit.spec.js:124`

**Impact:** Algorithm runs but results are invisible to user

**Findings:**
- Result message state exists (`result_message` signal)
- Output written to state but not rendered visibly
- No clear visual area for results

**Fix Required:**
- Add prominent results panel
- Style with background, padding, clear typography
- Show:
  - Success/error indicator
  - Algorithm output (cluster assignments, transformed data, etc.)
  - Execution time
  - Model performance metrics

---

## High Priority Bugs (P1 - Major UX Issues)

### BUG #5: No Train/Test Split Functionality
**Status:** ❌ CONFIRMED
**Severity:** HIGH
**Test:** `ml-playground-audit.spec.js:168`

**Issue:** No UI for splitting data into train/test sets

**Impact:**
- Cannot evaluate model generalization
- No way to see overfitting
- Not following ML best practices

**Fix Required:**
```rust
// Add train/test split UI
div { class: "train-test-split",
    h3 { "Data Split" }

    label { "Train Percentage:" }
    input {
        r#type: "range",
        min: "50",
        max: "90",
        value: "{train_split_pct}",
        oninput: move |evt| {
            train_split_pct.set(evt.value().parse().unwrap_or(80));
        }
    }
    span { "{train_split_pct}% train / {100 - train_split_pct}% test" }
}
```

**Algorithm Changes:**
```rust
// Split data before training
let split_idx = (n_samples as f64 * (train_pct / 100.0)) as usize;
let train_X = &X.rows(0..split_idx);
let test_X = &X.rows(split_idx..n_samples);

// Train on train_X, evaluate on test_X
model.fit(train_X, train_y)?;
let train_preds = model.predict(train_X)?;
let test_preds = model.predict(test_X)?;

// Show both train and test metrics
```

---

### BUG #6: No Predictions Table or Visualization
**Status:** ❌ CONFIRMED
**Severity:** HIGH
**Test:** `ml-playground-audit.spec.js:179`

**Issue:** No table showing actual predictions

**Impact:** Users can't see what the model predicted for each sample

**Fix Required:**
```rust
// Add predictions table
div { class: "predictions-container",
    h3 { "Predictions" }

    table { class: "predictions-table",
        thead {
            tr {
                th { "Sample" }
                th { "Actual" }
                th { "Predicted" }
                th { "Correct?" }
            }
        }
        tbody {
            for (i, (&actual, &pred)) in test_y.iter().zip(predictions.iter()).enumerate() {
                tr {
                    class: if actual == pred as f64 { "correct" } else { "incorrect" },
                    td { "{i + 1}" }
                    td { "{actual}" }
                    td { "{pred}" }
                    td { if actual == pred as f64 { "✓" } else { "✗" } }
                }
            }
        }
    }
}
```

---

### BUG #7: No Summary Statistics Display
**Status:** ❌ CONFIRMED
**Severity:** HIGH
**Test:** `ml-playground-audit.spec.js:190`

**Issue:** No accuracy, confusion matrix, or performance metrics shown

**Impact:** Users have no idea how well the model performs

**Fix Required:**
```rust
// Add summary statistics card
div { class: "summary-stats",
    h3 { "Model Performance" }

    div { class: "stat-row",
        span { class: "stat-label", "Accuracy:" }
        span { class: "stat-value", "{accuracy:.2}%" }
    }

    div { class: "stat-row",
        span { class: "stat-label", "Precision:" }
        span { class: "stat-value", "{precision:.2}" }
    }

    div { class: "stat-row",
        span { class: "stat-label", "Recall:" }
        span { class: "stat-value", "{recall:.2}" }
    }

    // Confusion Matrix
    h4 { "Confusion Matrix" }
    table { class: "confusion-matrix",
        // Render 2D confusion matrix
    }
}
```

---

### BUG #8: Parameter Sliders Don't Work
**Status:** ⚠️ PARTIALLY CONFIRMED
**Severity:** HIGH
**Context:** From user report and code review

**Issue from Code Review (ml_playground.rs:232):**
```rust
// WRONG - AlgorithmConfigurator sends "n_clusters" but we check for "k"
"k" => if let Some(val) = param.current_value.as_i64() {
    current_params.k_clusters = val as usize;
},
```

**This was FIXED in commit `0b8a1f0` (Nov 8):**
```rust
// CORRECT
"n_clusters" => if let Some(val) = param.current_value.as_i64() {
    current_params.k_clusters = val as usize;
},
```

**However, sliders still may not appear or may not update correctly.**

**Test Results:**
- Sliders might exist but not be visible
- Parameter changes might not trigger algorithm re-run
- No real-time feedback on parameter changes

**Fix Required:**
1. Ensure AlgorithmConfigurator renders for selected algorithm
2. Wire parameter changes to state
3. Show current parameter values clearly
4. Allow instant re-run with new parameters

---

## Medium Priority Issues (P2 - Nice to Have)

### Issue #9: Poor Visual Hierarchy
**Impact:** Hard to understand what to do next

**Fix:** Clear step-by-step UI:
```
Step 1: Upload Data → Step 2: Configure → Step 3: Train → Step 4: Evaluate
```

---

### Issue #10: No Progress Indicators
**Impact:** User doesn't know if algorithm is running

**Fix:** Add loading spinner and progress bar

---

### Issue #11: No Export Functionality
**Impact:** Can't save results or model

**Fix:** Add "Export Predictions" and "Export Model" buttons

---

## Redesign Proposal

### New ML Playground Structure

```
┌─────────────────────────────────────────────────────────────┐
│                        ML PLAYGROUND                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  STEP 1: UPLOAD DATA                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ [Choose CSV File] iris.csv uploaded ✓                │   │
│  │ Samples: 150 | Features: 4                           │   │
│  │                                                       │   │
│  │ Train/Test Split: ▓▓▓▓▓▓▓▓░░  80% / 20%            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  STEP 2: SELECT & CONFIGURE ALGORITHM                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Algorithm: [K-Means ▼]                               │   │
│  │                                                       │   │
│  │ Number of Clusters (k): ▓▓▓░░░░░ 3                 │   │
│  │ Max Iterations:         ▓▓▓▓▓▓░░ 100                │   │
│  │ Tolerance:              1e-4                          │   │
│  │                                                       │   │
│  │ [Run K-Means]                                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  STEP 3: RESULTS                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ✅ K-Means completed in 127ms                        │   │
│  │                                                       │   │
│  │ TRAINING SET (120 samples)                           │   │
│  │ ├─ Accuracy: 94.2%                                   │   │
│  │ └─ Inertia: 78.85                                    │   │
│  │                                                       │   │
│  │ TEST SET (30 samples)                                │   │
│  │ ├─ Accuracy: 93.3%                                   │   │
│  │ └─ Inertia: 19.23                                    │   │
│  │                                                       │   │
│  │ [View Predictions Table] [View Confusion Matrix]     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  PREDICTIONS TABLE                                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Sample │ Actual │ Predicted │ Correct?               │   │
│  │────────┼────────┼───────────┼────────────           │   │
│  │    1   │   0    │     0     │    ✓                   │   │
│  │    2   │   0    │     0     │    ✓                   │   │
│  │    3   │   1    │     2     │    ✗                   │   │
│  │   ...  │  ...   │    ...    │   ...                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Priority

### Phase 1: Make It Work (P0 Bugs)
1. Add algorithm selection dropdown
2. Add "Run Algorithm" button
3. Fix CSV upload visibility
4. Display results prominently
5. Fix parameter wiring (already done in commit `0b8a1f0`)

### Phase 2: Make It Useful (P1 Bugs)
6. Add train/test split
7. Show predictions table
8. Display summary statistics
9. Make parameter sliders actually work

### Phase 3: Make It Great (P2)
10. Add visual hierarchy
11. Progress indicators
12. Export functionality
13. Algorithm comparison
14. Interactive visualizations

---

## Testing Strategy

All bugs found via Playwright tests in `web/tests/ml-playground-audit.spec.js`:

```bash
# Run full audit
npx playwright test ml-playground-audit.spec.js

# Run specific bug test
npx playwright test ml-playground-audit.spec.js -g "BUG #1"

# Generate screenshots
npx playwright test --headed
```

**Test Coverage:**
- ✅ Navigation
- ✅ CSV upload
- ✅ Algorithm selection (FAILS)
- ✅ Parameter configuration (FAILS)
- ✅ Algorithm execution (FAILS)
- ✅ Results display (FAILS)
- ✅ Full workflow (FAILS)

---

## Success Criteria

ML Playground will be considered **fixed** when:

1. ✅ User can upload CSV
2. ✅ User can select algorithm from dropdown
3. ✅ User can configure parameters with sliders
4. ✅ User can click "Run Algorithm" button
5. ✅ Results appear in clear, prominent panel
6. ✅ Predictions table shows individual predictions
7. ✅ Summary statistics show accuracy/metrics
8. ✅ Train/test split allows evaluation

**All Playwright tests must pass.**

---

## Next Steps

1. Fix P0 bugs (algorithm selection, run button, results display)
2. Add Playwright tests for each fix
3. Implement train/test split
4. Add predictions visualization
5. Add summary statistics
6. Polish and deploy

**Estimated Time:** 4-6 hours for Phase 1+2

---

**Report Generated:** November 9, 2025
**Tools Used:** Playwright E2E Testing
**Test File:** `web/tests/ml-playground-audit.spec.js`
**Screenshots:** `web/test-results/ml-playground-audit-*/`
