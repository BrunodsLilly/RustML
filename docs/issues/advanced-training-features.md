# feat: Advanced Training Features for Linear Regression

## Overview

Enhance the linear regression trainer with professional ML features: cross-validation, train/test splits, multiple regression algorithms (Ridge, Lasso), and hyperparameter tuning. Transform the trainer from educational demo to production-ready ML workbench.

## Problem Statement

The current linear regression implementation (from PR #3) provides basic gradient descent training but lacks:
- **Model validation:** No way to assess generalization performance
- **Overfitting detection:** No train/test split or cross-validation
- **Regularization:** No Ridge or Lasso for high-dimensional data
- **Hyperparameter optimization:** Manual tuning of learning rate, iterations
- **Algorithm comparison:** Can't compare different regression approaches

**Impact:** Users can train models but can't evaluate them properly, detect overfitting, or choose optimal hyperparameters. This limits real-world applicability and educational value.

## Proposed Solution

Add production ML capabilities while maintaining the platform's client-side, zero-backend philosophy:

1. **Train/Test Split** - Automatic dataset partitioning
2. **K-Fold Cross-Validation** - Robust performance estimation
3. **Regularized Regression** - Ridge (L2) and Lasso (L1) implementations
4. **Hyperparameter Tuning** - Grid search and random search
5. **Algorithm Comparison** - Side-by-side performance metrics

## Technical Approach

### Architecture

**New Module Structure:**
```
linear_regression/src/
├── lib.rs                    (EXISTING - basic gradient descent)
├── validation.rs             (NEW - train/test split, k-fold CV)
├── regularized.rs            (NEW - Ridge, Lasso implementations)
├── hyperparameter_tuning.rs  (NEW - grid search, random search)
└── metrics.rs                (NEW - R², MAE, RMSE, etc.)

web/src/components/
├── training_config.rs        (NEW - algorithm & param selection)
├── validation_results.rs     (NEW - display CV scores)
└── showcase.rs              (MODIFY - integrate new features)
```

**Data Flow:**
```
CsvDataset → TrainingConfig (algorithm, params, validation strategy)
    ↓
ValidationStrategy
├─→ TrainTestSplit → single train/test evaluation
└─→ KFoldCV → K evaluations → aggregate metrics
    ↓
SelectedAlgorithm (OLS, Ridge, Lasso)
    ↓
TrainedModel + ValidationMetrics
    ↓
ResultsDisplay (coefficients, scores, plots)
```

### Implementation Phases

#### Phase 1: Train/Test Split & Metrics (Week 1, 12-16 hours)

**Tasks:**
- [ ] Implement dataset splitting in `linear_regression/src/validation.rs`
  - Random split with seed for reproducibility
  - Stratified split for regression (optional)
  - Shuffle before split
  - Validate split ratio (0.1-0.5)
- [ ] Create evaluation metrics in `linear_regression/src/metrics.rs`
  - R² (coefficient of determination)
  - MSE (mean squared error)
  - MAE (mean absolute error)
  - RMSE (root mean squared error)
  - MAPE (mean absolute percentage error)
- [ ] Add validation to existing `LinearRegressor`
  - `.evaluate(X_test, y_test) -> Metrics` method
  - Store both train and test metrics
- [ ] Create `TrainingResults` component in web
  - Side-by-side train vs test scores
  - Visual indicators for overfitting (train >> test)
  - Score interpretation tooltips
- [ ] Add unit tests for splitting and metrics
- [ ] Add E2E test: upload CSV, split, train, verify test scores

**Files to Create:**
```rust
// linear_regression/src/validation.rs
pub struct TrainTestSplit {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
    pub split_ratio: f64,
}

impl TrainTestSplit {
    pub fn new(n_samples: usize, test_ratio: f64, seed: Option<u64>) -> Self

    pub fn split_dataset(&self, X: &Matrix<f64>, y: &[f64])
        -> (Matrix<f64>, Vec<f64>, Matrix<f64>, Vec<f64>)
}

// linear_regression/src/metrics.rs
#[derive(Debug, Clone)]
pub struct RegressionMetrics {
    pub r_squared: f64,
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub mape: f64,
}

impl RegressionMetrics {
    pub fn calculate(y_true: &[f64], y_pred: &[f64]) -> Self

    pub fn interpret_r_squared(&self) -> &str {
        match self.r_squared {
            r if r > 0.9 => "Excellent fit",
            r if r > 0.7 => "Good fit",
            r if r > 0.5 => "Moderate fit",
            _ => "Poor fit",
        }
    }
}

// linear_regression/src/lib.rs (additions)
impl LinearRegressor {
    pub fn evaluate(&self, X: &Matrix<f64>, y: &[f64]) -> RegressionMetrics {
        let predictions = self.predict(X).unwrap();
        RegressionMetrics::calculate(y, &predictions)
    }
}
```

**Success Criteria:**
- ✅ Split produces correct sizes for any valid ratio
- ✅ All metrics calculated correctly (verified against sklearn)
- ✅ R² = 1.0 for perfect predictions, 0.0 for mean baseline
- ✅ Train/test metrics display updates immediately after training

#### Phase 2: K-Fold Cross-Validation (Week 2, 16-20 hours)

**Tasks:**
- [ ] Implement K-fold CV in `validation.rs`
  - Create K folds with equal sizes
  - Handle remainder samples (distribute evenly)
  - Support stratification (advanced feature)
  - Track fold indices for reproducibility
- [ ] Add cross-validation training loop
  - Train K models (one per fold)
  - Aggregate metrics across folds
  - Calculate mean and standard deviation
  - Store individual fold results
- [ ] Create async training with progress updates
  - Use `use_resource` for async CV
  - Update UI after each fold completes
  - Show "Fold 3/5 training..." status
  - Cancel training if user navigates away
- [ ] Create `CrossValidationResults` component
  - Display mean ± std for each metric
  - Show individual fold scores in expandable table
  - Visualize fold variance with error bars
  - Highlight best and worst folds
- [ ] Add tests for CV logic
- [ ] Add E2E test: run 5-fold CV, verify 5 models trained

**Files to Create:**
```rust
// linear_regression/src/validation.rs (continued)
pub struct KFoldCV {
    pub n_folds: usize,
    pub shuffle: bool,
    pub random_seed: Option<u64>,
}

impl KFoldCV {
    pub fn new(n_folds: usize) -> Self

    pub fn split_indices(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        // Returns (train_indices, val_indices) for each fold
    }

    pub fn cross_validate<M: Model>(
        &self,
        model_factory: impl Fn() -> M,
        X: &Matrix<f64>,
        y: &[f64],
    ) -> CVResults
}

#[derive(Debug, Clone)]
pub struct CVResults {
    pub fold_metrics: Vec<RegressionMetrics>,
    pub mean_metrics: RegressionMetrics,
    pub std_metrics: RegressionMetrics,
}

impl CVResults {
    pub fn is_overfitting(&self, threshold: f64) -> bool {
        // Check if std deviation is high relative to mean
        self.std_metrics.r_squared / self.mean_metrics.r_squared > threshold
    }
}
```

**Web Component:**
```rust
// web/src/components/cross_validation_results.rs
#[component]
pub fn CrossValidationResults(cv_results: CVResults) -> Element {
    rsx! {
        div { class: "cv-results",
            h3 { "Cross-Validation Results" }

            // Summary metrics
            MetricsSummary { results: cv_results.clone() }

            // Individual folds
            details {
                summary { "View Individual Folds" }
                FoldsTable { folds: cv_results.fold_metrics }
            }

            // Variance visualization
            ErrorBarChart {
                mean: cv_results.mean_metrics,
                std: cv_results.std_metrics,
            }
        }
    }
}
```

**Success Criteria:**
- ✅ K-fold produces exactly K train/val splits with no data leakage
- ✅ Each sample appears in validation set exactly once
- ✅ CV completes in <5 seconds for 1k samples, 5 folds
- ✅ Progress updates smoothly during async training
- ✅ Mean ± std displayed with clear interpretation

#### Phase 3: Regularized Regression (Ridge & Lasso) (Week 3, 20-24 hours)

**Tasks:**
- [ ] Implement Ridge Regression (L2 regularization)
  - Closed-form solution: w = (X^T X + λI)^(-1) X^T y
  - Gradient descent with L2 penalty (alternative method)
  - Automatic scaling of features
  - Handle matrix inversion edge cases
- [ ] Implement Lasso Regression (L1 regularization)
  - Coordinate descent algorithm
  - Soft thresholding operator
  - Feature selection via sparsity
  - Early stopping on convergence
- [ ] Add hyperparameter: regularization strength (λ/alpha)
  - Valid range: 0.0001 to 100.0 (log scale)
  - Default: 1.0
  - UI: Slider with log scale
- [ ] Create unified `Regressor` enum
  - `OLS` (ordinary least squares - existing)
  - `Ridge { alpha: f64 }`
  - `Lasso { alpha: f64, max_iter: usize }`
- [ ] Update UI to select algorithm
  - Radio buttons or dropdown
  - Show algorithm description
  - Conditional params (alpha only for Ridge/Lasso)
- [ ] Add tests comparing to sklearn
- [ ] Add E2E test: train all three algorithms, compare results

**Files to Create:**
```rust
// linear_regression/src/regularized.rs
pub struct RidgeRegressor {
    pub weights: Vector<f64>,
    pub bias: f64,
    pub alpha: f64,  // L2 penalty strength
    pub training_history: Vec<f64>,
}

impl RidgeRegressor {
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0, "Alpha must be positive");
        Self {
            weights: Vector { data: vec![] },
            bias: 0.0,
            alpha,
            training_history: vec![],
        }
    }

    pub fn fit_closed_form(&mut self, X: &Matrix<f64>, y: &[f64]) -> Result<(), String> {
        // Closed-form solution: w = (X^T X + αI)^(-1) X^T y
        let XtX = X.transpose().multiply(X)?;
        let regularization = Matrix::identity(XtX.cols).scale(self.alpha);
        let XtX_reg = XtX.add(&regularization)?;

        // Solve linear system (requires matrix inversion)
        let inverse = XtX_reg.inverse()?;
        let Xty = X.transpose().multiply_vector(&Vector { data: y.to_vec() })?;
        self.weights = inverse.multiply_vector(&Xty)?;

        Ok(())
    }

    pub fn fit_gradient_descent(&mut self, X: &Matrix<f64>, y: &[f64], iterations: usize) {
        // Alternative: gradient descent with L2 penalty
        for _ in 0..iterations {
            let predictions = self.predict(X).unwrap();
            let errors = subtract_vectors(&predictions, y);

            // Gradient with L2 penalty: X^T (predictions - y) + 2α w
            let mut gradient = X.transpose().multiply_vector(&Vector { data: errors }).unwrap();
            let penalty = self.weights.scale(2.0 * self.alpha);
            gradient = gradient.add(&penalty).unwrap();

            self.weights = self.weights.subtract(&gradient.scale(self.learning_rate)).unwrap();

            // Track cost
            let cost = self.cost(&predictions, y) + self.alpha * self.weights.squared_norm();
            self.training_history.push(cost);
        }
    }
}

pub struct LassoRegressor {
    pub weights: Vector<f64>,
    pub bias: f64,
    pub alpha: f64,  // L1 penalty strength
    pub max_iter: usize,
    pub tolerance: f64,
    pub training_history: Vec<f64>,
}

impl LassoRegressor {
    pub fn fit(&mut self, X: &Matrix<f64>, y: &[f64]) -> Result<(), String> {
        // Coordinate descent algorithm
        let n_features = X.cols;

        for iter in 0..self.max_iter {
            let weights_old = self.weights.clone();

            // Update each weight coordinate-wise
            for j in 0..n_features {
                let X_j = X.get_column(j);

                // Compute residual without feature j
                let residual = self.compute_residual_excluding(X, y, j);

                // Soft thresholding operator
                let rho_j = X_j.dot(&residual);
                self.weights.data[j] = soft_threshold(rho_j, self.alpha);
            }

            // Check convergence
            let diff = self.weights.subtract(&weights_old).unwrap().norm();
            if diff < self.tolerance {
                break;
            }

            // Track cost
            let predictions = self.predict(X)?;
            let cost = self.cost(&predictions, y) + self.alpha * self.weights.l1_norm();
            self.training_history.push(cost);
        }

        Ok(())
    }
}

fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0  // Sparsity: set to zero if within threshold
    }
}

// Unified interface
pub enum RegressorType {
    OLS(LinearRegressor),
    Ridge(RidgeRegressor),
    Lasso(LassoRegressor),
}

impl RegressorType {
    pub fn fit(&mut self, X: &Matrix<f64>, y: &[f64]) -> Result<(), String> {
        match self {
            RegressorType::OLS(model) => { model.fit(X, &Vector { data: y.to_vec() }, 1000); Ok(()) }
            RegressorType::Ridge(model) => model.fit_closed_form(X, y),
            RegressorType::Lasso(model) => model.fit(X, y),
        }
    }
}
```

**Web Component Updates:**
```rust
// web/src/components/training_config.rs
#[component]
pub fn TrainingConfig(on_config_change: EventHandler<TrainingConfiguration>) -> Element {
    let mut algorithm = use_signal(|| AlgorithmType::OLS);
    let mut alpha = use_signal(|| 1.0);

    rsx! {
        div { class: "training-config",
            h3 { "Algorithm Selection" }

            // Algorithm picker
            select {
                value: "{algorithm}",
                onchange: move |e| algorithm.set(parse_algorithm(&e.value())),
                option { value: "ols", "Ordinary Least Squares" }
                option { value: "ridge", "Ridge (L2 Regularization)" }
                option { value: "lasso", "Lasso (L1 Regularization)" }
            }

            // Algorithm description
            p { class: "algorithm-desc",
                {match algorithm() {
                    AlgorithmType::OLS => "Standard linear regression with no regularization",
                    AlgorithmType::Ridge => "Adds L2 penalty to prevent overfitting. Good for correlated features.",
                    AlgorithmType::Lasso => "Adds L1 penalty for feature selection. Creates sparse models.",
                }}
            }

            // Conditional regularization strength
            if algorithm() != AlgorithmType::OLS {
                div { class: "hyperparameter",
                    label { "Regularization Strength (α)" }
                    input {
                        r#type: "range",
                        min: "-4",
                        max: "2",
                        step: "0.1",
                        value: "{alpha().log10()}",
                        oninput: move |e| alpha.set(10f64.powf(e.value().parse().unwrap())),
                    }
                    span { "{alpha():.4}" }
                    p { class: "hint",
                        "Higher values = stronger regularization = simpler models"
                    }
                }
            }
        }
    }
}
```

**Success Criteria:**
- ✅ Ridge produces same results as sklearn with tolerance 1e-6
- ✅ Lasso selects features (some weights exactly 0.0)
- ✅ Both algorithms handle multicollinearity better than OLS
- ✅ Training completes in <1 second for 1k samples, 20 features
- ✅ UI updates smoothly when changing algorithm

#### Phase 4: Hyperparameter Tuning (Week 4, 16-20 hours)

**Tasks:**
- [ ] Implement grid search in `hyperparameter_tuning.rs`
  - Exhaustive search over parameter grid
  - Parallel evaluation using fold results
  - Track all combinations and scores
  - Return best parameters
- [ ] Implement random search (faster alternative)
  - Sample N random configurations
  - Configurable budget (number of trials)
  - Log-uniform sampling for alpha
- [ ] Add tuning UI component
  - Define parameter ranges
  - Choose search strategy (grid vs random)
  - Display progress: "Trial 15/100..."
  - Show results table sorted by score
- [ ] Integrate with cross-validation
  - Nested CV: outer loop for evaluation, inner for tuning
  - Avoid data leakage
  - Report generalization estimate
- [ ] Add async execution with cancellation
- [ ] Add tests for tuning logic
- [ ] Add E2E test: run grid search, verify best params selected

**Files to Create:**
```rust
// linear_regression/src/hyperparameter_tuning.rs
pub struct GridSearch {
    pub param_grid: HashMap<String, Vec<f64>>,
    pub cv_folds: usize,
}

impl GridSearch {
    pub fn fit<F>(&self,
        model_factory: F,
        X: &Matrix<f64>,
        y: &[f64],
    ) -> TuningResults
    where F: Fn(&HashMap<String, f64>) -> RegressorType
    {
        let mut results = vec![];

        // Generate all combinations
        for params in self.generate_combinations() {
            let model = model_factory(&params);
            let cv = KFoldCV::new(self.cv_folds);
            let cv_results = cv.cross_validate(|| model.clone(), X, y);

            results.push(TrialResult {
                params,
                mean_score: cv_results.mean_metrics.r_squared,
                std_score: cv_results.std_metrics.r_squared,
            });
        }

        // Sort by score
        results.sort_by(|a, b| b.mean_score.partial_cmp(&a.mean_score).unwrap());

        TuningResults {
            trials: results,
            best_params: results[0].params.clone(),
            best_score: results[0].mean_score,
        }
    }
}

pub struct RandomSearch {
    pub param_distributions: HashMap<String, ParamDistribution>,
    pub n_iter: usize,
    pub cv_folds: usize,
    pub random_seed: Option<u64>,
}

pub enum ParamDistribution {
    LogUniform { low: f64, high: f64 },
    Uniform { low: f64, high: f64 },
    Choice { values: Vec<f64> },
}

impl RandomSearch {
    pub fn sample_params(&self) -> HashMap<String, f64> {
        // Sample from distributions
    }
}
```

**Web Component:**
```rust
// web/src/components/hyperparameter_tuning_ui.rs
#[component]
pub fn HyperparameterTuningUI(
    dataset: CsvDataset,
    on_tuning_complete: EventHandler<TuningResults>,
) -> Element {
    let mut tuning_progress = use_signal(|| None::<(usize, usize)>);

    let start_tuning = use_resource(move || async move {
        // Run grid search asynchronously
        let grid_search = GridSearch {
            param_grid: hashmap! {
                "alpha" => vec![0.01, 0.1, 1.0, 10.0, 100.0],
                "learning_rate" => vec![0.001, 0.01, 0.1],
            },
            cv_folds: 5,
        };

        grid_search.fit(|params| {
            // Update progress
            tuning_progress.set(Some((current_trial, total_trials)));

            // Create model with params
            RegressorType::Ridge(RidgeRegressor::new(params["alpha"]))
        }, &dataset.features, &dataset.targets)
    });

    rsx! {
        div { class: "hyperparameter-tuning",
            h3 { "Hyperparameter Tuning" }

            if let Some((current, total)) = tuning_progress() {
                ProgressBar { current, total }
            }

            if let Some(Ok(results)) = start_tuning.value() {
                TuningResultsTable { results }
            }
        }
    }
}
```

**Success Criteria:**
- ✅ Grid search explores all combinations correctly
- ✅ Random search samples from distributions uniformly
- ✅ Best parameters match manual tuning results
- ✅ Progress updates smoothly during search
- ✅ Tuning completes in <30 seconds for 25 trials, 5-fold CV

## Alternative Approaches Considered

### 1. Use Python Backend for Training (Rejected)
**Pros:** Leverage scikit-learn, faster development
**Cons:** Breaks client-side philosophy, privacy concerns, network latency
**Decision:** Keep pure Rust/WASM

### 2. Bayesian Optimization (Future Enhancement)
**Pros:** More efficient than grid/random search
**Cons:** Complex implementation, requires GP library
**Decision:** Start with grid/random, add Bayesian later

### 3. Elastic Net (Ridge + Lasso) (Future Enhancement)
**Pros:** Best of both regularizations
**Cons:** More complex tuning (two hyperparameters)
**Decision:** Add after Ridge/Lasso proven

## Acceptance Criteria

### Functional Requirements

- [ ] **Train/Test Split**
  - Splits dataset with configurable ratio (0.1-0.5)
  - Shuffles before splitting
  - Reproducible with seed

- [ ] **Evaluation Metrics**
  - Calculates R², MSE, MAE, RMSE, MAPE correctly
  - Matches sklearn results within 1e-6
  - Displays train vs test scores

- [ ] **Cross-Validation**
  - Supports 3-10 folds
  - Each sample in validation exactly once
  - Returns mean ± std across folds

- [ ] **Ridge Regression**
  - Closed-form solution for exact results
  - Gradient descent alternative
  - Handles multicollinearity

- [ ] **Lasso Regression**
  - Coordinate descent algorithm
  - Produces sparse weights (feature selection)
  - Converges within max iterations

- [ ] **Hyperparameter Tuning**
  - Grid search over parameter ranges
  - Random search for faster exploration
  - Returns best parameters and scores

### Non-Functional Requirements

- [ ] **Performance**
  - Train/test split: <10ms for 10k samples
  - 5-fold CV: <5s for 1k samples, 10 features
  - Ridge fit: <100ms for 1k samples, 20 features
  - Lasso fit: <500ms for 1k samples, 20 features
  - Grid search: <30s for 25 trials, 5-fold CV

- [ ] **Accuracy**
  - All algorithms match sklearn within 1e-5
  - Metrics match reference implementations
  - No numerical instability for well-conditioned problems

- [ ] **Memory**
  - Bounded memory during CV (no accumulation)
  - Efficient fold storage (indices not data copies)
  - <100 MB total for largest datasets

### Quality Gates

- [ ] **Test Coverage**
  - Unit tests for all algorithms (>95% coverage)
  - Integration tests for validation pipeline
  - E2E tests for complete workflow
  - Benchmark tests comparing to sklearn

- [ ] **Code Quality**
  - Clear documentation with examples
  - No panics in WASM code paths
  - Bounded iterations and memory
  - Performance notes for large datasets

## Success Metrics

**Educational Impact:**
- Users understand overfitting (train vs test scores)
- 60%+ try cross-validation
- 40%+ experiment with regularization

**Technical Performance:**
- <5s total time for full validation workflow
- Zero crashes during extended sessions
- Memory stable over 100+ training runs

**Engagement:**
- Average time exploring validation features: >3 minutes
- 30%+ run hyperparameter tuning
- Positive feedback on "production-ready" feel

## Dependencies & Prerequisites

**Required:**
- ✅ Linear regression implementation (existing)
- ✅ Matrix operations library (existing)
- ✅ CSV upload integration (PR #3)

**New:**
- Matrix inversion function for Ridge closed-form
- Async resource handling in Dioxus

## Risk Analysis & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Matrix inversion unstable | High | Medium | Use SVD-based pseudo-inverse, add conditioning checks |
| Lasso doesn't converge | Medium | Low | Implement early stopping, warn if max_iter reached |
| CV too slow for large datasets | Medium | Medium | Add progress indicator, allow cancellation, limit to 10 folds |
| Hyperparameter tuning expensive | Low | High | Default to random search (faster), limit trials |

## Resource Requirements

**Time:** 4 weeks (64-80 hours)
- Week 1: Train/test + metrics (12-16h)
- Week 2: Cross-validation (16-20h)
- Week 3: Ridge + Lasso (20-24h)
- Week 4: Hyperparameter tuning (16-20h)

**Team:** 1 developer with ML + Rust experience

## Future Considerations

- Elastic Net regression
- Bayesian hyperparameter optimization
- Learning curves for sample size analysis
- Feature scaling/normalization UI
- Model persistence (save/load)

## References

### Internal
- Linear Regression: `linear_regression/src/lib.rs`
- Matrix Operations: `linear_algebra/src/lib.rs`
- Async Patterns: `web/src/components/optimizer_demo.rs`

### External
- [Ridge Regression](https://en.wikipedia.org/wiki/Ridge_regression)
- [Lasso Regression](https://en.wikipedia.org/wiki/Lasso_(statistics))
- [Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Hyperparameter Tuning](https://scikit-learn.org/stable/modules/grid_search.html)

---

**Labels:** `enhancement`, `machine-learning`, `algorithms`, `advanced`

**Estimated Effort:** Large (4 weeks)

**Priority:** P2 (High Value, Medium Complexity)
