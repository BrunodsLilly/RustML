# feat: User Experience Improvements for CSV Upload & Training

## Overview

Polish the CSV upload and linear regression training workflow with professional UX features: model export/import, batch prediction interface, real-time loss visualization, and better error recovery. Transform functional features into delightful experiences.

## Problem Statement

The current CSV upload and training integration (from PR #3) works but has UX gaps:
- **No persistence:** Trained models lost on page refresh
- **Limited prediction:** Can't make batch predictions on new data
- **Static training:** No real-time feedback during training
- **Poor error recovery:** Upload errors require page refresh
- **Missing workflows:** Can't compare multiple models, export results

**Impact:** Users invest time training models but can't save work, make predictions efficiently, or recover from errors gracefully. This reduces engagement and limits practical utility.

## Proposed Solution

Add professional UX polish to make the platform production-ready:

1. **Model Export/Import** - Save and load trained models
2. **Batch Prediction Interface** - Upload new data for predictions
3. **Real-Time Loss Chart** - Live visualization during training
4. **Error Recovery** - Graceful handling with retry mechanisms
5. **Model Comparison** - Side-by-side evaluation of multiple models

## Technical Approach

### Architecture

**New Components:**
```
web/src/components/
├── model_export.rs           (NEW - download JSON/WASM binary)
├── model_import.rs           (NEW - upload and restore model)
├── batch_predictor.rs        (NEW - CSV upload for predictions)
├── live_loss_chart.rs        (NEW - SVG/Canvas real-time chart)
├── error_recovery_ui.rs      (NEW - retry mechanisms)
├── model_comparison.rs       (NEW - multi-model dashboard)
└── showcase.rs               (MODIFY - integrate all features)

linear_regression/src/
├── serialization.rs          (NEW - serde traits for save/load)
└── lib.rs                    (MODIFY - implement Serialize/Deserialize)
```

**Data Flow:**
```
TrainedModel → Serialize → JSON/Binary → Download
    ↓
Upload → Deserialize → Model → Predictions

NewDataCSV → Parse → Predict → Results CSV → Download
```

### Implementation Phases

#### Phase 1: Model Export/Import (Week 1, 12-16 hours)

**Tasks:**
- [ ] Add serde serialization to `LinearRegressor`
  - Derive `Serialize` and `Deserialize`
  - Include weights, bias, hyperparameters
  - Add model metadata (timestamp, feature names, metrics)
  - Version field for backwards compatibility
- [ ] Create export component
  - Download as JSON (human-readable)
  - Download as MessagePack (compact binary)
  - Include dataset schema for validation
  - Auto-generate filename with timestamp
- [ ] Create import component
  - Drag-and-drop file upload
  - Validate model format and version
  - Check feature compatibility with current dataset
  - Preview model details before loading
- [ ] Add browser storage option
  - Save to localStorage (limit 5 models)
  - List saved models with metadata
  - Delete old models
  - Auto-save latest model
- [ ] Add tests for serialization
- [ ] Add E2E test: export model, refresh page, import, verify same predictions

**Files to Create:**
```rust
// linear_regression/src/serialization.rs
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializedModel {
    pub version: String,  // "1.0.0" for backwards compatibility
    pub model_type: String,  // "LinearRegressor"
    pub weights: Vec<f64>,
    pub bias: f64,
    pub hyperparameters: ModelHyperparameters,
    pub metadata: ModelMetadata,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelHyperparameters {
    pub learning_rate: f64,
    pub iterations: usize,
    pub regularization: Option<RegularizationType>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelMetadata {
    pub created_at: String,  // ISO 8601 timestamp
    pub feature_names: Vec<String>,
    pub n_features: usize,
    pub n_samples_trained: usize,
    pub final_loss: f64,
    pub metrics: Option<RegressionMetrics>,
}

impl LinearRegressor {
    pub fn serialize(&self, metadata: ModelMetadata) -> SerializedModel {
        SerializedModel {
            version: "1.0.0".to_string(),
            model_type: "LinearRegressor".to_string(),
            weights: self.weights.data.clone(),
            bias: self.bias,
            hyperparameters: ModelHyperparameters {
                learning_rate: self.learning_rate,
                iterations: self.training_history.len(),
                regularization: None,
            },
            metadata,
        }
    }

    pub fn deserialize(model: &SerializedModel) -> Result<Self, String> {
        // Validate version
        if model.version != "1.0.0" {
            return Err(format!("Unsupported model version: {}", model.version));
        }

        // Reconstruct model
        Ok(LinearRegressor {
            weights: Vector { data: model.weights.clone() },
            bias: model.bias,
            learning_rate: model.hyperparameters.learning_rate,
            training_history: vec![],  // Don't restore history
        })
    }
}
```

**Web Component:**
```rust
// web/src/components/model_export.rs
#[component]
pub fn ModelExport(model: LinearRegressor, metadata: ModelMetadata) -> Element {
    let export_json = move |_| {
        let serialized = model.serialize(metadata.clone());
        let json = serde_json::to_string_pretty(&serialized).unwrap();

        // Create download link
        let filename = format!("linear_model_{}.json",
            chrono::Utc::now().format("%Y%m%d_%H%M%S"));

        download_file(&json, &filename, "application/json");
    };

    rsx! {
        div { class: "model-export",
            h3 { "Export Trained Model" }

            button {
                onclick: export_json,
                "📥 Download as JSON"
            }

            div { class: "model-info",
                p { "Features: {metadata.n_features}" }
                p { "Training samples: {metadata.n_samples_trained}" }
                p { "Final loss: {metadata.final_loss:.6}" }
            }
        }
    }
}

// web/src/components/model_import.rs
#[component]
pub fn ModelImport(on_import: EventHandler<LinearRegressor>) -> Element {
    let mut error = use_signal(|| None::<String>);

    let handle_file = move |evt: FormEvent| async move {
        let files = evt.files();
        if let Some(file) = files.first() {
            match file.read_text().await {
                Ok(content) => {
                    match serde_json::from_str::<SerializedModel>(&content) {
                        Ok(serialized) => {
                            match LinearRegressor::deserialize(&serialized) {
                                Ok(model) => {
                                    on_import.call(model);
                                    error.set(None);
                                }
                                Err(e) => error.set(Some(e)),
                            }
                        }
                        Err(e) => error.set(Some(format!("Invalid JSON: {}", e))),
                    }
                }
                Err(e) => error.set(Some(format!("Failed to read file: {}", e))),
            }
        }
    };

    rsx! {
        div { class: "model-import",
            h3 { "Import Trained Model" }

            input {
                r#type: "file",
                accept: ".json",
                onchange: handle_file,
            }

            if let Some(err) = error() {
                div { class: "error", "{err}" }
            }
        }
    }
}
```

**Success Criteria:**
- ✅ Exported JSON is valid and human-readable
- ✅ Imported model produces identical predictions to original
- ✅ File size <1 KB for typical model
- ✅ Import validates model version and feature compatibility
- ✅ localStorage stores up to 5 models without overflow

#### Phase 2: Batch Prediction Interface (Week 2, 12-16 hours)

**Tasks:**
- [ ] Create batch prediction component
  - Upload CSV with same features as training data
  - Validate schema matches (column names, count)
  - Make predictions for all rows
  - Display results in table
  - Download predictions as CSV
- [ ] Add prediction confidence/uncertainty
  - Prediction intervals (optional advanced feature)
  - Flag out-of-range inputs (extrapolation warning)
- [ ] Create results visualization
  - Scatter plot: actual vs predicted (if ground truth provided)
  - Histogram of predictions
  - Summary statistics (min, max, mean, std)
- [ ] Add tests for batch prediction
- [ ] Add E2E test: train model, upload new data, verify predictions

**Files to Create:**
```rust
// web/src/components/batch_predictor.rs
#[component]
pub fn BatchPredictor(
    model: LinearRegressor,
    feature_names: Vec<String>,
) -> Element {
    let mut predictions = use_signal(|| None::<Vec<PredictionResult>>);
    let mut error = use_signal(|| None::<String>);

    let handle_csv_upload = move |csv_content: String| {
        // Parse CSV
        match parse_prediction_csv(&csv_content, &feature_names) {
            Ok(features_matrix) => {
                // Make predictions
                match model.predict(&features_matrix) {
                    Ok(preds) => {
                        let results = preds.iter().enumerate()
                            .map(|(i, &pred)| PredictionResult {
                                row_index: i,
                                prediction: pred,
                                warning: check_extrapolation(&features_matrix, i),
                            })
                            .collect();
                        predictions.set(Some(results));
                        error.set(None);
                    }
                    Err(e) => error.set(Some(format!("Prediction failed: {}", e))),
                }
            }
            Err(e) => error.set(Some(e)),
        }
    };

    rsx! {
        div { class: "batch-predictor",
            h3 { "Batch Predictions" }

            CsvUploader {
                on_loaded: move |content| handle_csv_upload(content),
                expected_columns: feature_names.clone(),
            }

            if let Some(err) = error() {
                ErrorMessage { message: err }
            }

            if let Some(preds) = predictions() {
                div { class: "prediction-results",
                    PredictionTable { predictions: preds.clone() }
                    DownloadPredictionsButton { predictions: preds }
                }
            }
        }
    }
}

#[derive(Clone)]
struct PredictionResult {
    row_index: usize,
    prediction: f64,
    warning: Option<String>,  // e.g., "Extrapolation detected"
}

fn check_extrapolation(X: &Matrix<f64>, row: usize) -> Option<String> {
    // Check if any feature value is outside training data range
    // This requires storing training data min/max during model training
    None  // Simplified for now
}

fn download_predictions_csv(predictions: &[PredictionResult]) {
    let csv_content = format!(
        "row,prediction\n{}",
        predictions.iter()
            .map(|p| format!("{},{:.6}", p.row_index, p.prediction))
            .collect::<Vec<_>>()
            .join("\n")
    );

    download_file(&csv_content, "predictions.csv", "text/csv");
}
```

**Success Criteria:**
- ✅ Validates uploaded CSV has correct columns
- ✅ Makes predictions in <100ms for 10k rows
- ✅ Downloads CSV with predictions
- ✅ Shows warnings for out-of-range inputs
- ✅ Handles missing values gracefully

#### Phase 3: Real-Time Loss Visualization (Week 3, 16-20 hours)

**Tasks:**
- [ ] Modify training loop to emit progress events
  - Update every N iterations (configurable)
  - Send current iteration, loss, weights
  - Use callbacks or channels
- [ ] Create live loss chart component
  - SVG line chart for <1000 iterations
  - Canvas for >1000 iterations (performance)
  - Auto-scaling axes
  - Hover tooltips with exact values
  - Zoom and pan controls
- [ ] Add training progress UI
  - Progress bar with percentage
  - Time elapsed and ETA
  - Current loss value (large, readable)
  - Pause/resume training button
- [ ] Implement pause/resume mechanism
  - Store training state
  - Continue from current weights
  - Don't lose history
- [ ] Add tests for training callbacks
- [ ] Add E2E test: start training, verify chart updates in real-time

**Files to Modify:**
```rust
// linear_regression/src/lib.rs (additions)
pub type ProgressCallback = Box<dyn Fn(usize, f64, &Vector<f64>)>;

impl LinearRegressor {
    pub fn fit_with_progress(
        &mut self,
        X: &Matrix<f64>,
        y: &Vector<f64>,
        iterations: usize,
        progress_callback: Option<ProgressCallback>,
        update_every: usize,
    ) {
        for i in 0..iterations {
            // Existing training logic
            self.step(X, y);

            // Progress callback
            if let Some(ref callback) = progress_callback {
                if i % update_every == 0 || i == iterations - 1 {
                    let loss = self.training_history.last().copied().unwrap_or(0.0);
                    callback(i, loss, &self.weights);
                }
            }
        }
    }

    pub fn pause_training(&mut self) -> TrainingState {
        TrainingState {
            weights: self.weights.clone(),
            bias: self.bias,
            history: self.training_history.clone(),
        }
    }

    pub fn resume_training(&mut self, state: TrainingState, X: &Matrix<f64>, y: &Vector<f64>, iterations: usize) {
        self.weights = state.weights;
        self.bias = state.bias;
        self.training_history = state.history;

        self.fit(X, y, iterations);
    }
}
```

**Web Component:**
```rust
// web/src/components/live_loss_chart.rs
#[component]
pub fn LiveLossChart(loss_history: Vec<f64>) -> Element {
    let mut zoom_level = use_signal(|| 1.0);

    // Determine rendering method based on data size
    let use_canvas = loss_history.len() > 1000;

    rsx! {
        div { class: "live-loss-chart",
            h3 { "Training Loss" }

            div { class: "current-loss",
                span { class: "loss-value",
                    "{loss_history.last().unwrap_or(&0.0):.6}"
                }
                span { class: "loss-label", " MSE" }
            }

            if use_canvas {
                CanvasChart {
                    data: loss_history,
                    zoom: zoom_level(),
                }
            } else {
                SvgChart {
                    data: loss_history,
                    zoom: zoom_level(),
                }
            }

            div { class: "chart-controls",
                button { onclick: move |_| zoom_level *= 1.2, "Zoom In" }
                button { onclick: move |_| zoom_level /= 1.2, "Zoom Out" }
                button { onclick: move |_| zoom_level.set(1.0), "Reset" }
            }
        }
    }
}

// Training progress component
#[component]
pub fn TrainingProgress(
    current_iteration: usize,
    total_iterations: usize,
    elapsed_ms: u128,
) -> Element {
    let progress_pct = (current_iteration as f64 / total_iterations as f64 * 100.0);
    let eta_ms = if current_iteration > 0 {
        (elapsed_ms as f64 / current_iteration as f64) * (total_iterations - current_iteration) as f64
    } else {
        0.0
    };

    rsx! {
        div { class: "training-progress",
            div { class: "progress-bar",
                div {
                    class: "progress-fill",
                    style: "width: {progress_pct}%",
                }
            }

            div { class: "progress-stats",
                span { "Iteration: {current_iteration} / {total_iterations}" }
                span { "Elapsed: {elapsed_ms / 1000}s" }
                span { "ETA: {eta_ms / 1000:.1}s" }
            }
        }
    }
}
```

**Success Criteria:**
- ✅ Chart updates smoothly at 10-30 FPS during training
- ✅ No jank or frame drops
- ✅ Axes auto-scale to data range
- ✅ Pause/resume works without data loss
- ✅ Canvas rendering handles 100k+ iterations

#### Phase 4: Error Recovery & Model Comparison (Week 4, 12-16 hours)

**Tasks:**
- [ ] Improve error recovery UI
  - Retry button for failed operations
  - Clear error message with fix suggestions
  - Don't lose user's work (keep CSV, settings)
  - Undo last action
- [ ] Add model comparison dashboard
  - Train multiple models (different algorithms, hyperparams)
  - Side-by-side metrics table
  - Overlaid loss charts
  - Highlight best model
- [ ] Create model registry
  - Store up to 10 models in memory
  - Name each model
  - Delete unwanted models
  - Compare any 2-4 models
- [ ] Add tests for error recovery
- [ ] Add E2E test: trigger error, retry, verify success

**Files to Create:**
```rust
// web/src/components/error_recovery_ui.rs
#[component]
pub fn ErrorRecoveryUI(
    error: String,
    retry_action: EventHandler<()>,
    clear_action: EventHandler<()>,
) -> Element {
    rsx! {
        div { class: "error-recovery",
            div { class: "error-message",
                "⚠️ {error}"
            }

            div { class: "error-suggestions",
                "Suggestions:"
                ul {
                    {suggest_fixes(&error).iter().map(|fix| rsx! {
                        li { "{fix}" }
                    })}
                }
            }

            div { class: "error-actions",
                button {
                    onclick: move |_| retry_action.call(()),
                    "🔄 Retry"
                }
                button {
                    onclick: move |_| clear_action.call(()),
                    "✖ Clear Error"
                }
            }
        }
    }
}

fn suggest_fixes(error: &str) -> Vec<String> {
    let mut suggestions = vec![];

    if error.contains("column") {
        suggestions.push("Check that your CSV has the correct columns".to_string());
    }
    if error.contains("numeric") {
        suggestions.push("Ensure all values are valid numbers".to_string());
    }
    if error.contains("NaN") || error.contains("Infinity") {
        suggestions.push("Remove or replace non-finite values (NaN, Inf)".to_string());
    }

    suggestions
}

// web/src/components/model_comparison.rs
#[derive(Clone)]
struct ModelEntry {
    id: usize,
    name: String,
    model: LinearRegressor,
    metrics: RegressionMetrics,
    algorithm: String,
}

#[component]
pub fn ModelComparison(models: Vec<ModelEntry>) -> Element {
    let mut selected_models = use_signal(|| vec![0, 1]);  // Compare first 2 by default

    rsx! {
        div { class: "model-comparison",
            h3 { "Model Comparison" }

            // Model selector
            div { class: "model-selector",
                "Select models to compare:"
                for model in models.iter() {
                    label {
                        input {
                            r#type: "checkbox",
                            checked: selected_models().contains(&model.id),
                            onchange: move |_| toggle_selection(model.id, &mut selected_models),
                        }
                        "{model.name}"
                    }
                }
            }

            // Comparison table
            ComparisonTable {
                models: models.iter()
                    .filter(|m| selected_models().contains(&m.id))
                    .cloned()
                    .collect(),
            }

            // Overlaid loss charts
            OverlaidLossCharts {
                models: models.iter()
                    .filter(|m| selected_models().contains(&m.id))
                    .cloned()
                    .collect(),
            }
        }
    }
}
```

**Success Criteria:**
- ✅ Error recovery preserves user's work
- ✅ Retry mechanism works for all error types
- ✅ Suggestions are contextually relevant
- ✅ Can compare up to 4 models side-by-side
- ✅ Best model clearly highlighted

## Alternative Approaches Considered

### 1. Server-Side Model Storage (Rejected)
**Pros:** No localStorage limits, synced across devices
**Cons:** Breaks privacy promise, requires backend
**Decision:** Use localStorage + download for now

### 2. IndexedDB Instead of localStorage (Future)
**Pros:** Larger storage, better for binary data
**Cons:** More complex API
**Decision:** Start simple, migrate if needed

### 3. WebWorkers for Training (Future)
**Pros:** Non-blocking UI during training
**Cons:** Data transfer overhead, complexity
**Decision:** Add after basic real-time updates proven

## Acceptance Criteria

### Functional Requirements

- [ ] **Model Export/Import**
  - Serialize to JSON with metadata
  - Import validates version and features
  - localStorage stores 5 most recent models
  - Download generates timestamped filename

- [ ] **Batch Prediction**
  - Validates CSV schema matches training
  - Makes predictions for all rows
  - Downloads results as CSV
  - Warns about extrapolation

- [ ] **Real-Time Chart**
  - Updates during training (10-30 FPS)
  - Auto-scales axes
  - Supports zoom/pan
  - Uses Canvas for >1000 iterations

- [ ] **Error Recovery**
  - Shows contextual fix suggestions
  - Retry preserves user's work
  - Clear error messages
  - No data loss on error

- [ ] **Model Comparison**
  - Compare up to 4 models
  - Side-by-side metrics
  - Overlaid loss charts
  - Highlight best model

### Non-Functional Requirements

- [ ] **Performance**
  - Export: <100ms
  - Import: <200ms
  - Batch prediction: <100ms for 10k rows
  - Chart updates: 30 FPS minimum

- [ ] **Usability**
  - Drag-and-drop file upload
  - Keyboard shortcuts (Ctrl+S to export)
  - Responsive design (mobile friendly)
  - Accessibility (WCAG 2.1 AA)

- [ ] **Reliability**
  - No data loss on page refresh (localStorage)
  - Graceful degradation if localStorage full
  - Robust error handling

### Quality Gates

- [ ] **Test Coverage**
  - Unit tests for serialization (>95%)
  - Integration tests for workflows
  - E2E tests for all features
  - Error scenario tests

- [ ] **User Testing**
  - 5 users complete full workflow
  - Average SUS score >70
  - No critical usability issues

## Success Metrics

**Engagement:**
- 50%+ export at least one model
- 30%+ use batch prediction
- 70%+ interact with real-time chart

**Satisfaction:**
- Average session time increases 2x
- Positive feedback on "polish"
- NPS score >50

**Retention:**
- 40%+ return within 7 days
- 20%+ import previously saved model

## Dependencies & Prerequisites

**Required:**
- ✅ CSV upload (PR #3)
- ✅ Linear regression training (existing)
- ✅ Dioxus file handling (existing)

**New:**
- `serde` and `serde_json` for serialization
- `chrono` for timestamps (optional)

## Risk Analysis & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| localStorage quota exceeded | Medium | Medium | Show warning at 80%, compress models, limit to 5 |
| Real-time chart jank | High | Low | Use Canvas for large data, throttle updates |
| Import breaks on model changes | Medium | Medium | Version field, backwards compatibility checks |
| Users don't find export/import | Low | High | Prominent placement, onboarding tour |

## Resource Requirements

**Time:** 4 weeks (52-68 hours)
- Week 1: Export/import (12-16h)
- Week 2: Batch prediction (12-16h)
- Week 3: Real-time chart (16-20h)
- Week 4: Error recovery + comparison (12-16h)

**Team:** 1 developer + optional UX designer

## Future Considerations

- Cloud sync (optional backend)
- Model versioning and diff
- Collaborative model sharing
- Mobile app with model import
- Automated model selection

## References

### Internal
- CSV Upload: `web/src/components/csv_upload.rs`
- File Handling: `web/src/components/showcase.rs:606-616`
- Visualization: `web/src/components/optimizer_demo.rs`

### External
- [Serde Documentation](https://serde.rs/)
- [Canvas API](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)
- [localStorage](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage)
- [UX Best Practices for ML](https://pair.withgoogle.com/guidebook/)

---

**Labels:** `enhancement`, `ux`, `polish`, `user-experience`

**Estimated Effort:** Medium (4 weeks)

**Priority:** P2 (High Impact on Engagement)
