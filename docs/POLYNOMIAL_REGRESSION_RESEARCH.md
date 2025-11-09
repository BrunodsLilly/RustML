# Polynomial Regression in Rust + WASM: Research Summary

**Date:** November 8, 2025
**Purpose:** Comprehensive best practices for implementing polynomial regression with interactive WASM visualization

---

## Table of Contents

1. [Mathematical Foundations](#mathematical-foundations)
2. [Rust Implementation Patterns](#rust-implementation-patterns)
3. [Numerical Stability & Performance](#numerical-stability--performance)
4. [Regularization Techniques](#regularization-techniques)
5. [WASM Visualization Best Practices](#wasm-visualization-best-practices)
6. [Testing & Validation](#testing--validation)
7. [Educational Design Patterns](#educational-design-patterns)
8. [Recommended Architecture](#recommended-architecture)
9. [Code Examples](#code-examples)
10. [References](#references)

---

## 1. Mathematical Foundations

### 1.1 Polynomial Regression Theory

Polynomial regression extends linear regression using the polynomial form:

```
f(x) = a_n * x^n + a_(n-1) * x^(n-1) + ... + a_1 * x + a_0
```

**Key Concept:** Transform the problem into a linear system by creating polynomial features from input data.

### 1.2 Vandermonde Matrix Approach

The **Vandermonde matrix** is the standard method for polynomial regression:

```
V = [
  [1,  x₁,  x₁²,  ...,  x₁ⁿ]
  [1,  x₂,  x₂²,  ...,  x₂ⁿ]
  ...
  [1,  xₘ,  xₘ²,  ...,  xₘⁿ]
]
```

**Normal Equations:** Solve `V^T × V × a = V^T × y` for coefficients `a`

**Source:** Wikipedia, Vandermonde matrix | Rust ML implementations

### 1.3 Alternative Methods

1. **Singular Value Decomposition (SVD)** - Most numerically stable
2. **Gradient Descent** - For large datasets or regularized models
3. **QR Decomposition** - Good balance of stability and performance

**Recommendation:** Use SVD for numerical stability, especially with high-degree polynomials (degree > 5).

---

## 2. Rust Implementation Patterns

### 2.1 Core Libraries

**Primary:**
- `ndarray` - NumPy-like arrays for Rust
- `ndarray-linalg` - Linear algebra operations (SVD, solve)

**Alternative:**
- `nalgebra` - Full-featured linear algebra library
- `linfa` - Rust ML library (scikit-learn-like)

**Specialized:**
- `polynomial` - Polynomial manipulation
- `horner` - Efficient polynomial evaluation

### 2.2 Recommended Vandermonde Implementation

Based on community examples from Sling Academy and rust-ml.github.io:

```rust
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;

pub fn polynomial_fit(
    x: &Array1<f64>,
    y: &Array1<f64>,
    degree: usize
) -> Result<Array1<f64>, PolynomialError> {
    // Validate inputs
    if x.len() != y.len() {
        return Err(PolynomialError::DimensionMismatch);
    }
    if x.len() <= degree {
        return Err(PolynomialError::InsufficientData {
            points: x.len(),
            degree,
        });
    }

    // Construct Vandermonde matrix
    let n = x.len();
    let mut vandermonde = Array2::zeros((n, degree + 1));

    for (i, &xi) in x.iter().enumerate() {
        for j in 0..=degree {
            vandermonde[[i, j]] = xi.powi(j as i32);
        }
    }

    // Solve normal equations
    let vt = vandermonde.t();
    let coeffs = vt
        .dot(&vandermonde)
        .solve(&vt.dot(y))
        .map_err(|_| PolynomialError::SingularMatrix)?;

    Ok(coeffs)
}
```

### 2.3 Error Handling Pattern

**Best Practice:** Use custom error types instead of `unwrap()`

```rust
#[derive(Debug, Clone)]
pub enum PolynomialError {
    DimensionMismatch,
    InsufficientData { points: usize, degree: usize },
    SingularMatrix,
    InvalidDegree(usize),
    NumericalInstability,
}

impl std::fmt::Display for PolynomialError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::InsufficientData { points, degree } => write!(
                f,
                "Need at least {} points for degree {} polynomial, got {}",
                degree + 1, degree, points
            ),
            // ... other variants
        }
    }
}
```

**Source:** Rust ML Book patterns, adapted from linear regression examples

### 2.4 Trait Design for Extensibility

**Pattern:** Separate model representation from fitting algorithm

```rust
pub trait Regressor {
    type Params;

    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), RegressionError>;
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, RegressionError>;
    fn params(&self) -> &Self::Params;
}

pub struct PolynomialRegression {
    degree: usize,
    coefficients: Option<Array1<f64>>,
    feature_transformer: Box<dyn FeatureTransform>,
}

impl Regressor for PolynomialRegression {
    type Params = Array1<f64>;

    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), RegressionError> {
        // Transform to polynomial features
        let x_poly = self.feature_transformer.transform(x)?;

        // Fit using chosen method (SVD, normal equations, etc.)
        self.coefficients = Some(fit_coefficients(&x_poly, y)?);
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, RegressionError> {
        let coeffs = self.coefficients.as_ref()
            .ok_or(RegressionError::ModelNotFitted)?;

        let x_poly = self.feature_transformer.transform(x)?;
        Ok(x_poly.dot(coeffs))
    }

    fn params(&self) -> &Self::Params {
        self.coefficients.as_ref().unwrap()
    }
}
```

**Rationale:** Allows swapping feature transformers (polynomial, splines, interactions) and fitting algorithms (SVD, gradient descent) without changing client code.

---

## 3. Numerical Stability & Performance

### 3.1 Polynomial Evaluation: Horner's Method

**Problem:** Naive evaluation `a₀ + a₁x + a₂x² + ... + aₙxⁿ` requires O(n²) multiplications

**Solution:** Horner's method restructures to O(n) time

```rust
/// Efficient polynomial evaluation using Horner's method
pub fn evaluate_polynomial(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter()
        .rev()
        .fold(0.0, |acc, &coeff| acc * x + coeff)
}

// Example: For 3x² + 2x + 1, coeffs = [1.0, 2.0, 3.0]
// Computes: ((0*x + 3)*x + 2)*x + 1
```

**Performance:** 10-50x faster for high-degree polynomials (degree > 10)

**Sources:**
- Horner's method Wikipedia
- Sling Academy Rust implementation
- GeeksforGeeks complexity analysis

### 3.2 Numerical Stability Considerations

**High-Degree Polynomial Challenges:**

1. **Matrix Conditioning:** Vandermonde matrices become ill-conditioned as degree increases
2. **Overflow Risk:** x^n can overflow for large x or high n
3. **Cancellation Errors:** Subtraction of similar values loses precision

**Mitigation Strategies:**

```rust
pub const MAX_SAFE_DEGREE: usize = 15;

pub fn validate_degree(degree: usize) -> Result<(), PolynomialError> {
    if degree > MAX_SAFE_DEGREE {
        return Err(PolynomialError::InvalidDegree(degree));
    }
    Ok(())
}

// Feature scaling before polynomial transformation
pub fn scale_features(x: &Array1<f64>) -> (Array1<f64>, f64, f64) {
    let mean = x.mean().unwrap();
    let std = x.std(0.0);
    let x_scaled = (x - mean) / std;
    (x_scaled, mean, std)
}
```

**Recommended Limits:**
- **Degrees 1-5:** Generally safe, no special handling needed
- **Degrees 6-10:** Scale features to [-1, 1] range
- **Degrees 11-15:** Use SVD instead of normal equations, scale features
- **Degrees 16+:** Consider alternatives (splines, piecewise polynomials)

**Source:** Fast_polynomial crate documentation, numerical analysis literature

### 3.3 Zero-Allocation Hot Paths (WASM Critical)

**Pattern:** Specialize for common visualization cases

```rust
impl PolynomialRegression {
    /// Zero-allocation prediction for single point (common in visualization loops)
    pub fn predict_single(&self, x: f64) -> f64 {
        let coeffs = self.coefficients.as_ref().expect("Model not fitted");
        evaluate_polynomial(coeffs.as_slice().unwrap(), x)
    }

    /// Batch prediction with pre-allocated output
    pub fn predict_batch_into(&self, x: &[f64], output: &mut [f64]) {
        let coeffs = self.coefficients.as_ref().expect("Model not fitted");
        for (i, &xi) in x.iter().enumerate() {
            output[i] = evaluate_polynomial(coeffs.as_slice().unwrap(), xi);
        }
    }
}
```

**Rationale:** Eliminates allocations in render loops (60 FPS requires <16ms per frame)

**Source:** Adapted from your optimizer zero-allocation pattern in `/neural_network/src/optimizer.rs:536-601`

---

## 4. Regularization Techniques

### 4.1 Ridge Regression (L2 Regularization)

**Purpose:** Prevent overfitting by penalizing large coefficients

**Cost Function:**
```
Cost = MSE + α * Σ(aᵢ²)
```

**Closed-Form Solution:**
```
a = (V^T × V + α × I)^(-1) × V^T × y
```

**Implementation:**

```rust
pub struct RidgeRegression {
    degree: usize,
    alpha: f64,  // Regularization strength
    coefficients: Option<Array1<f64>>,
}

impl RidgeRegression {
    pub fn new(degree: usize, alpha: f64) -> Result<Self, PolynomialError> {
        if alpha < 0.0 || !alpha.is_finite() {
            return Err(PolynomialError::InvalidRegularization);
        }
        Ok(Self { degree, alpha, coefficients: None })
    }

    pub fn fit(&mut self, x: &Array1<f64>, y: &Array1<f64>) -> Result<(), PolynomialError> {
        let vandermonde = create_vandermonde(x, self.degree);
        let vt = vandermonde.t();
        let n_features = self.degree + 1;

        // Add regularization term: V^T V + α I
        let mut gram = vt.dot(&vandermonde);
        for i in 0..n_features {
            gram[[i, i]] += self.alpha;
        }

        self.coefficients = Some(
            gram.solve(&vt.dot(y))
                .map_err(|_| PolynomialError::SingularMatrix)?
        );
        Ok(())
    }
}
```

**Hyperparameter Selection:** Use cross-validation to find optimal α

**Sources:**
- Scikit-learn Ridge documentation
- TowardsDataScience regularization guide

### 4.2 Feature Scaling Requirement

**Critical:** Regularization is scale-dependent

```rust
pub fn standardize_features(x: &Array1<f64>) -> (Array1<f64>, f64, f64) {
    let mean = x.mean().unwrap();
    let std = x.std(0.0);
    let x_scaled = (x - mean) / std;
    (x_scaled, mean, std)
}
```

**Best Practice:** Always scale features before applying Ridge/Lasso

**Rationale:** Without scaling, features with larger magnitudes dominate the penalty term, leading to unfair regularization.

**Source:** TowardsDataScience "Avoid This Pitfall" article

### 4.3 Lasso Regression (L1 Regularization)

**Purpose:** Feature selection + overfitting prevention

**Key Difference from Ridge:** Can zero out coefficients (automatic feature selection)

**Note:** No closed-form solution; requires iterative optimization (coordinate descent, proximal gradient)

**Implementation Complexity:** Higher than Ridge; consider using existing library (linfa) or implementing coordinate descent

**Source:** Medium regularization comparison articles

---

## 5. WASM Visualization Best Practices

### 5.1 Performance Targets

Based on your existing optimizer visualizer:

| Metric | Target | Critical For |
|--------|--------|--------------|
| Frame Rate | 60 FPS | Smooth animations |
| Curve Evaluation | 1000+ points/frame | High-resolution curves |
| Memory Growth | 0 (bounded) | Long-running demos |
| Allocations in render loop | 0 | WASM performance |

**Sources:**
- Your `/docs/PERFORMANCE_BENCHMARK.md`
- WASM visualization research (Orchestra guides)

### 5.2 Rendering Strategy: SVG vs Canvas

**SVG (Current Approach):**
- ✅ Declarative, easy to debug
- ✅ Built-in interactivity (hover, click)
- ❌ Performance degrades with >500 points
- ❌ Re-renders entire DOM on updates

**Canvas (Recommended for Polynomial Curves):**
- ✅ Constant-time rendering (1000+ points at 60 FPS)
- ✅ No DOM overhead
- ❌ Manual event handling for interactivity
- ❌ More complex code

**Recommendation:** Use Canvas for curve rendering, SVG for UI elements (axes, labels, controls)

**Source:** WASM data graph performance guides (getorchestra.io)

### 5.3 Efficient Curve Rendering Pattern

```rust
pub struct CurveRenderer {
    canvas: HtmlCanvasElement,
    ctx: CanvasRenderingContext2d,
    point_buffer: Vec<(f64, f64)>,  // Reused buffer
}

impl CurveRenderer {
    const BUFFER_SIZE: usize = 1000;

    pub fn new(canvas: HtmlCanvasElement) -> Self {
        let ctx = canvas.get_context("2d").unwrap();
        let point_buffer = Vec::with_capacity(Self::BUFFER_SIZE);
        Self { canvas, ctx, point_buffer }
    }

    pub fn render_polynomial(&mut self, model: &PolynomialRegression, x_min: f64, x_max: f64) {
        self.point_buffer.clear();

        // Generate curve points (zero allocations)
        let step = (x_max - x_min) / (Self::BUFFER_SIZE as f64 - 1.0);
        for i in 0..Self::BUFFER_SIZE {
            let x = x_min + i as f64 * step;
            let y = model.predict_single(x);
            self.point_buffer.push((x, y));
        }

        // Render to canvas
        self.ctx.begin_path();
        if let Some(&(x0, y0)) = self.point_buffer.first() {
            self.ctx.move_to(x0, y0);
            for &(x, y) in self.point_buffer.iter().skip(1) {
                self.ctx.line_to(x, y);
            }
        }
        self.ctx.stroke();
    }
}
```

**Key Optimizations:**
1. Pre-allocated buffer (no Vec growth)
2. `predict_single()` uses Horner's method (zero allocations)
3. Direct Canvas API calls (no intermediate structures)

### 5.4 Bounded Memory for Interactive Demos

**Pattern:** Circular buffers for history tracking

```rust
pub struct PolynomialTrainer {
    model: PolynomialRegression,
    cost_history: VecDeque<f64>,
    max_history: usize,
}

impl PolynomialTrainer {
    const MAX_HISTORY: usize = 1000;

    pub fn record_cost(&mut self, cost: f64) {
        if self.cost_history.len() >= Self::MAX_HISTORY {
            self.cost_history.pop_front();
        }
        self.cost_history.push_back(cost);
    }
}
```

**Rationale:** Prevents unbounded memory growth in long-running browser demos

**Source:** Your optimizer circular buffer pattern

---

## 6. Testing & Validation

### 6.1 Essential Test Cases

**Category 1: Exact Fits (Known Solutions)**

```rust
#[test]
fn test_linear_is_polynomial_degree_1() {
    // A degree-1 polynomial should exactly match linear regression
    let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]); // y = 2x

    let model = PolynomialRegression::new(1).unwrap();
    model.fit(&x, &y).unwrap();

    let coeffs = model.coefficients();
    assert_approx_eq!(coeffs[0], 0.0, 1e-10); // Intercept
    assert_approx_eq!(coeffs[1], 2.0, 1e-10); // Slope
}

#[test]
fn test_quadratic_exact_fit() {
    // Generate exact quadratic: y = 3x² + 2x + 1
    let x = Array1::linspace(-5.0, 5.0, 20);
    let y = x.mapv(|xi| 3.0 * xi.powi(2) + 2.0 * xi + 1.0);

    let model = PolynomialRegression::new(2).unwrap();
    model.fit(&x, &y).unwrap();

    let coeffs = model.coefficients();
    assert_approx_eq!(coeffs[0], 1.0, 1e-8);
    assert_approx_eq!(coeffs[1], 2.0, 1e-8);
    assert_approx_eq!(coeffs[2], 3.0, 1e-8);
}
```

**Category 2: Edge Cases**

```rust
#[test]
fn test_insufficient_data_points() {
    let x = Array1::from_vec(vec![1.0, 2.0]); // Only 2 points
    let y = Array1::from_vec(vec![1.0, 4.0]);

    let model = PolynomialRegression::new(3); // Degree 3 needs 4+ points
    assert!(model.fit(&x, &y).is_err());
}

#[test]
fn test_degree_zero_constant_fit() {
    let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = Array1::from_vec(vec![5.0, 5.0, 5.0, 5.0, 5.0]);

    let model = PolynomialRegression::new(0).unwrap();
    model.fit(&x, &y).unwrap();

    assert_approx_eq!(model.coefficients()[0], 5.0, 1e-10);
}

#[test]
fn test_high_degree_numerical_stability() {
    // High degree should still work with scaled features
    let x = Array1::linspace(-1.0, 1.0, 100); // Scaled range
    let y = x.mapv(|xi| xi.sin()); // Smooth target

    let model = PolynomialRegression::new(12).unwrap();
    assert!(model.fit(&x, &y).is_ok());

    // Check condition number or residuals to verify stability
}
```

**Category 3: Overfitting Detection**

```rust
#[test]
fn test_overfitting_behavior() {
    // Generate noisy data from quadratic
    let x_train = Array1::linspace(0.0, 10.0, 20);
    let y_train = x_train.mapv(|xi| xi.powi(2) + random_noise());

    let x_test = Array1::linspace(0.0, 10.0, 100);
    let y_test = x_test.mapv(|xi| xi.powi(2)); // No noise

    // Degree 2 should generalize well
    let model_good = PolynomialRegression::new(2).unwrap();
    model_good.fit(&x_train, &y_train).unwrap();
    let mse_good = compute_mse(&model_good, &x_test, &y_test);

    // Degree 15 should overfit
    let model_overfit = PolynomialRegression::new(15).unwrap();
    model_overfit.fit(&x_train, &y_train).unwrap();
    let mse_overfit = compute_mse(&model_overfit, &x_test, &y_test);

    assert!(mse_good < mse_overfit,
        "Lower degree should generalize better");
}
```

**Source:** STAT 501 polynomial regression examples, CS109A validation patterns

### 6.2 Validation Metrics

**Implementation:**

```rust
pub fn compute_r_squared(model: &PolynomialRegression, x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let y_pred = model.predict(x);
    let y_mean = y.mean().unwrap();

    let ss_res: f64 = y.iter()
        .zip(y_pred.iter())
        .map(|(yi, yi_pred)| (yi - yi_pred).powi(2))
        .sum();

    let ss_tot: f64 = y.iter()
        .map(|yi| (yi - y_mean).powi(2))
        .sum();

    1.0 - (ss_res / ss_tot)
}

pub fn compute_adjusted_r_squared(r_squared: f64, n_samples: usize, n_features: usize) -> f64 {
    let n = n_samples as f64;
    let p = n_features as f64;
    1.0 - (1.0 - r_squared) * (n - 1.0) / (n - p - 1.0)
}

pub fn compute_rmse(model: &PolynomialRegression, x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let y_pred = model.predict(x);
    let mse: f64 = y.iter()
        .zip(y_pred.iter())
        .map(|(yi, yi_pred)| (yi - yi_pred).powi(2))
        .sum::<f64>() / y.len() as f64;
    mse.sqrt()
}
```

**Display in UI:**

```rust
pub struct ModelMetrics {
    pub r_squared: f64,
    pub adjusted_r_squared: f64,
    pub rmse: f64,
    pub degree: usize,
}

impl ModelMetrics {
    pub fn quality_badge(&self) -> &str {
        match self.r_squared {
            r if r >= 0.95 => "Excellent",
            r if r >= 0.85 => "Good",
            r if r >= 0.70 => "Fair",
            _ => "Poor",
        }
    }
}
```

**Source:** Medium regression evaluation articles, STAT 501 validation metrics

### 6.3 Cross-Validation for Degree Selection

```rust
pub fn select_optimal_degree(
    x: &Array1<f64>,
    y: &Array1<f64>,
    max_degree: usize,
    k_folds: usize
) -> usize {
    let mut best_degree = 1;
    let mut best_score = f64::NEG_INFINITY;

    for degree in 1..=max_degree {
        let cv_scores = k_fold_cross_validation(x, y, degree, k_folds);
        let mean_score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;

        if mean_score > best_score {
            best_score = mean_score;
            best_degree = degree;
        }
    }

    best_degree
}
```

**Source:** Harvard CS109A model selection lectures, Python Data Science Handbook

---

## 7. Educational Design Patterns

### 7.1 Interactive Parameter Controls

**Based on bias-variance playground analysis:**

**Essential Controls:**
1. **Polynomial Degree Slider** (1-15)
   - Real-time curve update
   - Visual feedback of overfitting/underfitting

2. **Regularization Strength** (0.0-10.0)
   - Show coefficient shrinkage effect
   - Compare regularized vs unregularized

3. **Data Noise Level** (0%-50%)
   - Demonstrate robustness
   - Show when high degree fails

4. **Sample Size** (10-200 points)
   - Illustrate data requirements
   - Show overfitting with small samples

**Implementation Pattern:**

```rust
#[component]
pub fn PolynomialPlayground(cx: Scope) -> Element {
    let degree = use_state(cx, || 2usize);
    let alpha = use_state(cx, || 0.0f64);
    let noise = use_state(cx, || 0.1f64);
    let n_samples = use_state(cx, || 50usize);

    // Auto-retrain on parameter change
    let model = use_memo(cx, (degree, alpha, noise, n_samples),
        |(deg, alph, nse, n)| {
            let data = generate_noisy_data(*n, *nse);
            train_model(&data, *deg, *alph)
        }
    );

    render! {
        div {
            ParameterControls {
                degree: degree,
                alpha: alpha,
                noise: noise,
                n_samples: n_samples,
            }
            PolynomialCurvePlot { model: model }
            BiasVarianceChart { model: model }
            MetricsDisplay { model: model }
        }
    }
}
```

### 7.2 Multi-View Visualization Strategy

**View 1: Main Scatter + Fitted Curve**
- Data points (with noise visualization)
- Polynomial curve overlay
- Residual lines (optional toggle)

**View 2: Bias-Variance Decomposition**
- Box plot showing variance across multiple fits
- Bias bars showing systematic error
- Interactive legend explaining tradeoff

**View 3: Coefficient Display**
- Bar chart of learned coefficients
- Highlight coefficient magnitude changes with regularization
- Show equation form

**View 4: Cross-Validation Plot**
- Train vs validation error curves
- Optimal degree marker
- Overfitting/underfitting regions

**Source:** D3 bias-variance playground, MLU-Explain interactive design

### 7.3 Progressive Disclosure Pattern

**Level 1: Beginner (Default)**
- Simple degree slider
- Single scatter plot
- Basic metrics (R²)

**Level 2: Intermediate**
- Regularization controls
- Multiple datasets
- Train/test split visualization

**Level 3: Advanced**
- Custom loss functions
- Manual coefficient adjustment
- Export trained models

**Implementation:**

```rust
#[derive(Clone, Copy, PartialEq)]
pub enum ExpertiseLevel {
    Beginner,
    Intermediate,
    Advanced,
}

pub fn render_controls(level: ExpertiseLevel) -> Element {
    match level {
        ExpertiseLevel::Beginner => render! {
            DegreeSlider {}
            DatasetSelector {}
        },
        ExpertiseLevel::Intermediate => render! {
            DegreeSlider {}
            RegularizationControl {}
            TrainTestSplit {}
            DatasetSelector {}
        },
        ExpertiseLevel::Advanced => render! {
            AllControls {}
            CustomLossEditor {}
            ModelExporter {}
        },
    }
}
```

### 7.4 Explanatory Tooltips & Annotations

**Pattern:** Contextual help that teaches without blocking

```rust
pub struct TooltipContent {
    pub title: &'static str,
    pub explanation: &'static str,
    pub example: Option<&'static str>,
}

pub const DEGREE_TOOLTIP: TooltipContent = TooltipContent {
    title: "Polynomial Degree",
    explanation: "Controls model complexity. Higher degrees fit training data better but may overfit.",
    example: Some("Degree 1 = straight line, Degree 2 = parabola"),
};

pub const REGULARIZATION_TOOLTIP: TooltipContent = TooltipContent {
    title: "Regularization (α)",
    explanation: "Penalizes large coefficients to prevent overfitting. Higher values = simpler models.",
    example: Some("Try α=0.1 to reduce oscillations in high-degree polynomials"),
};
```

**Visual Indicators:**
- Color-code overfitting regions (red) vs optimal (green)
- Animated transitions when parameters change
- Highlight which coefficient changes most

**Source:** D3 playground interactive features, Observable visualization patterns

---

## 8. Recommended Architecture

### 8.1 Crate Structure

```
polynomial_regression/
├─ src/
│  ├─ lib.rs                    # Public API
│  ├─ model.rs                  # PolynomialRegression struct
│  ├─ features.rs               # Vandermonde matrix creation
│  ├─ solvers/
│  │  ├─ mod.rs
│  │  ├─ normal_equations.rs    # V^T V solution
│  │  ├─ svd.rs                 # SVD-based solver
│  │  └─ gradient_descent.rs    # Iterative solver
│  ├─ regularization/
│  │  ├─ mod.rs
│  │  ├─ ridge.rs               # L2 regularization
│  │  └─ lasso.rs               # L1 regularization
│  ├─ evaluation/
│  │  ├─ mod.rs
│  │  ├─ horner.rs              # Fast polynomial evaluation
│  │  └─ metrics.rs             # R², RMSE, etc.
│  ├─ validation/
│  │  ├─ mod.rs
│  │  └─ cross_validation.rs    # K-fold CV
│  └─ errors.rs                 # Error types
├─ tests/
│  ├─ exact_fits.rs             # Known solution tests
│  ├─ edge_cases.rs             # Boundary conditions
│  └─ numerical_stability.rs    # High-degree tests
└─ examples/
   ├─ basic_fit.rs              # Simple example
   └─ cross_validation.rs       # Degree selection
```

### 8.2 Web Component Structure

```
web/src/components/
├─ polynomial_playground.rs     # Main container component
├─ polynomial_viz/
│  ├─ mod.rs
│  ├─ curve_plot.rs             # Canvas-based curve rendering
│  ├─ scatter_plot.rs           # SVG data points
│  ├─ residual_plot.rs          # Residual visualization
│  └─ bias_variance_chart.rs    # Dual metric display
├─ controls/
│  ├─ degree_slider.rs          # Degree selection
│  ├─ regularization_control.rs # Alpha parameter
│  └─ dataset_selector.rs       # Pre-loaded datasets
├─ metrics/
│  ├─ coefficient_display.rs    # Reuse from linear regression
│  └─ model_metrics.rs          # R², RMSE, etc.
└─ education/
   ├─ tooltip.rs                # Contextual help
   └─ onboarding_tour.rs        # First-time guide
```

### 8.3 Trait System for Extensibility

```rust
// Core regression trait
pub trait Regressor {
    type Params;
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), RegressionError>;
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, RegressionError>;
    fn params(&self) -> &Self::Params;
}

// Feature transformation trait
pub trait FeatureTransform {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, TransformError>;
    fn inverse_transform(&self, x_transformed: &Array2<f64>) -> Result<Array2<f64>, TransformError>;
}

// Polynomial feature transformer
pub struct PolynomialFeatures {
    degree: usize,
    include_bias: bool,
}

impl FeatureTransform for PolynomialFeatures {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, TransformError> {
        // Convert [x] to [1, x, x², ..., x^degree]
    }
}

// Solver trait for algorithm flexibility
pub trait RegressionSolver {
    fn solve(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>, SolverError>;
}

pub struct SVDSolver;
pub struct NormalEquationsSolver;
pub struct GradientDescentSolver { learning_rate: f64, max_iter: usize };

// Compose them
pub struct PolynomialRegression {
    feature_transform: Box<dyn FeatureTransform>,
    solver: Box<dyn RegressionSolver>,
    coefficients: Option<Array1<f64>>,
}
```

**Rationale:** Enables:
- Swapping solvers without changing client code
- Adding new feature transforms (splines, interactions)
- Testing different algorithms easily
- Reusing common components

**Source:** Rust ML Book patterns, linfa design philosophy

---

## 9. Code Examples

### 9.1 Complete Working Example

```rust
use ndarray::{Array1, Array2};
use polynomial_regression::{PolynomialRegression, RidgeRegression};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate synthetic data: y = 2x² + 3x + 1 + noise
    let x = Array1::linspace(-5.0, 5.0, 50);
    let y_true = x.mapv(|xi| 2.0 * xi.powi(2) + 3.0 * xi + 1.0);
    let noise = Array1::from_shape_fn(50, |_| rand::random::<f64>() * 2.0 - 1.0);
    let y = &y_true + &noise;

    // Fit polynomial regression
    let mut model = PolynomialRegression::new(2)?;
    model.fit(&x.insert_axis(ndarray::Axis(1)), &y)?;

    println!("Coefficients: {:?}", model.coefficients());

    // Evaluate on test data
    let x_test = Array1::linspace(-5.0, 5.0, 100);
    let y_pred = model.predict(&x_test.insert_axis(ndarray::Axis(1)))?;

    // Compute metrics
    let r_squared = compute_r_squared(&model, &x_test.insert_axis(ndarray::Axis(1)), &y_true);
    let rmse = compute_rmse(&model, &x.insert_axis(ndarray::Axis(1)), &y);

    println!("R² = {:.4}", r_squared);
    println!("RMSE = {:.4}", rmse);

    // Compare with Ridge regression
    let mut ridge = RidgeRegression::new(2, 0.1)?;
    ridge.fit(&x.insert_axis(ndarray::Axis(1)), &y)?;
    println!("Ridge coefficients: {:?}", ridge.coefficients());

    Ok(())
}
```

### 9.2 WASM Visualization Example

```rust
use dioxus::prelude::*;
use polynomial_regression::PolynomialRegression;

#[component]
pub fn PolynomialDemo(cx: Scope) -> Element {
    let degree = use_state(cx, || 2usize);
    let data = use_state(cx, || generate_sample_data());

    let model = use_memo(cx, (degree, data), |(deg, dat)| {
        let mut model = PolynomialRegression::new(*deg).unwrap();
        model.fit(&dat.x, &dat.y).unwrap();
        model
    });

    render! {
        div { class: "polynomial-demo",
            h2 { "Polynomial Regression Playground" }

            div { class: "controls",
                label { "Degree: {degree}" }
                input {
                    r#type: "range",
                    min: 1,
                    max: 10,
                    value: "{degree}",
                    oninput: move |evt| degree.set(evt.value.parse().unwrap()),
                }
            }

            PolynomialPlot {
                model: model.clone(),
                data: data.clone(),
            }

            ModelMetrics {
                model: model.clone(),
                data: data.clone(),
            }
        }
    }
}

#[component]
fn PolynomialPlot(cx: Scope, model: PolynomialRegression, data: DataSet) -> Element {
    let curve_points = use_memo(cx, model, |model| {
        // Generate 200 points for smooth curve
        let x_range = (-5.0..=5.0).step_by(0.05);
        x_range.map(|x| {
            let y = model.predict_single(x);
            (x, y)
        }).collect::<Vec<_>>()
    });

    render! {
        svg { width: "600", height: "400", viewBox: "0 0 600 400",
            // Axes
            line { x1: 50, y1: 350, x2: 550, y2: 350, stroke: "#333", stroke_width: 2 }
            line { x1: 50, y1: 50, x2: 50, y2: 350, stroke: "#333", stroke_width: 2 }

            // Data points
            for (x, y) in data.iter() {
                circle {
                    cx: scale_x(*x),
                    cy: scale_y(*y),
                    r: 4,
                    fill: "#6366f1",
                    opacity: 0.6,
                }
            }

            // Polynomial curve
            polyline {
                points: curve_points.iter()
                    .map(|(x, y)| format!("{},{}", scale_x(*x), scale_y(*y)))
                    .collect::<Vec<_>>()
                    .join(" "),
                fill: "none",
                stroke: "#ec4899",
                stroke_width: 2,
            }
        }
    }
}
```

### 9.3 Interactive Bias-Variance Demonstration

```rust
#[component]
pub fn BiasVarianceDemo(cx: Scope) -> Element {
    let degree = use_state(cx, || 2usize);
    let n_trials = 20; // Number of random samplings

    // Generate multiple models from different samples
    let models = use_memo(cx, degree, |deg| {
        (0..n_trials).map(|_| {
            let data = generate_noisy_data(50);
            let mut model = PolynomialRegression::new(*deg).unwrap();
            model.fit(&data.x, &data.y).unwrap();
            model
        }).collect::<Vec<_>>()
    });

    // Compute bias and variance
    let (bias, variance) = compute_bias_variance(&models, &true_function);

    render! {
        div { class: "bias-variance-demo",
            h3 { "Bias-Variance Tradeoff (Degree: {degree})" }

            div { class: "metrics-grid",
                MetricCard { title: "Bias²", value: bias, color: "#f59e0b" }
                MetricCard { title: "Variance", value: variance, color: "#8b5cf6" }
                MetricCard {
                    title: "Total Error",
                    value: bias + variance,
                    color: "#ef4444"
                }
            }

            // Show all model predictions as faint lines
            svg { width: "600", height: "400",
                for model in models.iter() {
                    PolynomialCurve {
                        model: model.clone(),
                        stroke: "#6366f1",
                        opacity: 0.1,
                    }
                }

                // True function
                TrueFunctionCurve { stroke: "#10b981", stroke_width: 3 }
            }

            BiasVarianceExplanation { degree: *degree }
        }
    }
}

fn compute_bias_variance(
    models: &[PolynomialRegression],
    true_fn: &dyn Fn(f64) -> f64
) -> (f64, f64) {
    let test_points = Array1::linspace(-5.0, 5.0, 100);

    let mut bias_sq = 0.0;
    let mut variance = 0.0;

    for &x in test_points.iter() {
        let true_y = true_fn(x);

        // Predictions from all models
        let predictions: Vec<f64> = models.iter()
            .map(|m| m.predict_single(x))
            .collect();

        let mean_pred = predictions.iter().sum::<f64>() / predictions.len() as f64;

        // Bias = (mean prediction - true value)²
        bias_sq += (mean_pred - true_y).powi(2);

        // Variance = E[(prediction - mean prediction)²]
        variance += predictions.iter()
            .map(|&pred| (pred - mean_pred).powi(2))
            .sum::<f64>() / predictions.len() as f64;
    }

    bias_sq /= test_points.len() as f64;
    variance /= test_points.len() as f64;

    (bias_sq, variance)
}
```

---

## 10. References

### Official Documentation
- [ndarray documentation](https://docs.rs/ndarray)
- [ndarray-linalg documentation](https://docs.rs/ndarray-linalg)
- [Horner crate](https://docs.rs/horner)
- [Scikit-learn Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

### Academic Sources
- Wikipedia: Vandermonde Matrix
- Wikipedia: Horner's Method
- Wikipedia: Bias-Variance Tradeoff
- STAT 501: Polynomial Regression Examples (Penn State)
- Harvard CS109A: Model Selection and Cross Validation

### Implementation Guides
- [Sling Academy: Rust Polynomial Curve Fitting](https://www.slingacademy.com/article/creating-a-rust-program-for-polynomial-curve-fitting/)
- [Rust ML Book: Linear Regression](https://rust-ml.github.io/book/5_linear_regression.html)
- [Medium: Regularization in Linear Regression](https://medium.com/@maxwienandts/regularization-in-linear-regression-a-deep-dive-into-ridge-and-lasso-3d2853e5e2b0)
- [TowardsDataScience: Avoid Lasso/Ridge Pitfalls](https://towardsdatascience.com/avoid-this-pitfall-when-using-lasso-and-ridge-regression-f4f4948bfe70/)

### WASM Performance
- [Orchestra: High-Performance Data Graphs with WASM](https://www.getorchestra.io/guides/high-performance-interactive-data-graphs-with-wasm)
- [Casey Primozic: Speeding Up Webcola with WASM](https://cprimozic.net/blog/speeding-up-webcola-with-webassembly/)

### Interactive Visualizations
- [D3 Bias-Variance Playground](https://gursimar.github.io/d3-visualizations/bias-var/)
- [MLU-Explain: Bias Variance](https://mlu-explain.github.io/bias-variance/)
- [Observable: Linear Regression Visualization](https://observablehq.com/@yizhe-ang/interactive-visualization-of-linear-regression)

### Rust Ecosystem
- [KDnuggets: Building ML Models in Rust](https://www.kdnuggets.com/building-high-performance-machine-learning-models-rust)
- [GeeksforGeeks: Horner's Method](https://www.geeksforgeeks.org/dsa/horners-method-polynomial-evaluation/)

---

## Summary: Key Takeaways

### Must-Have Features
1. **Vandermonde matrix** approach with SVD solver for numerical stability
2. **Horner's method** for efficient polynomial evaluation
3. **Ridge regularization** with proper feature scaling
4. **Comprehensive metrics**: R², Adjusted R², RMSE
5. **Cross-validation** for degree selection

### Performance Optimizations
1. Zero-allocation `predict_single()` for visualization loops
2. Pre-allocated buffers for curve rendering
3. Canvas rendering for >500 points
4. Bounded memory with circular buffers

### Educational Excellence
1. Interactive degree slider with real-time feedback
2. Bias-variance decomposition visualization
3. Multiple fitted curves showing variance
4. Contextual tooltips explaining concepts
5. Progressive disclosure (beginner → advanced modes)

### Testing Strategy
1. Exact fit tests (known polynomials)
2. Edge cases (insufficient data, degree 0, high degrees)
3. Overfitting detection (train vs test error)
4. Numerical stability tests (scaled features, condition numbers)

### Recommended Architecture
- Trait-based design (Regressor, FeatureTransform, Solver)
- Separate crate for core ML (`polynomial_regression`)
- Web components for interactive visualization
- Reusable evaluation and metrics modules

---

**Next Steps for Implementation:**

1. Create `polynomial_regression` crate with trait system
2. Implement Vandermonde + SVD solver
3. Add Horner's method evaluation
4. Build Ridge regression variant
5. Create interactive WASM playground component
6. Implement bias-variance visualization
7. Add comprehensive test suite
8. Write educational tooltips and onboarding

**Estimated Effort:** 2-3 weeks for full implementation with polished visualization

---

*Research compiled: November 8, 2025*
*For: brunoml/cargo_workspace*
*Revolutionary ML in the browser 🚀*
