# feat: Add L1/L2 Regularization to Neural Network

## Overview

Extend the existing `neural_network` crate with L1 and L2 regularization to prevent overfitting and improve generalization. This is a quick win that demonstrates professional ML practices and provides excellent educational value through interactive visualization.

**Priority:** 🟡 HIGH - Quick implementation (1-2 weeks), significant educational and practical impact.

## Problem Statement

### Current Limitations
- Neural networks can overfit on small datasets (memorize training data)
- No mechanism to prevent large weights
- Cannot demonstrate regularization effects in browser demos
- Missing industry-standard technique for model robustness

### Why This Matters
1. **Real-World ML:** Regularization is essential for production models
2. **Educational:** Perfect visualization opportunity (show overfitting vs regularization)
3. **Quick Win:** Extends existing code with minimal effort (1-2 weeks vs 4 weeks for CNN)
4. **Showcase:** Interactive demo proves understanding better than static tutorials

### User Stories
- **As a student:** I want to see how L2 regularization prevents overfitting on a small dataset
- **As a developer:** I want to train robust neural networks that generalize to unseen data
- **As an educator:** I want to show students the effect of different regularization strengths (lambda values)

## Proposed Solution

### High-Level Design

Add regularization as optional parameter to `NeuralNetwork`:

```rust
// neural_network/src/lib.rs
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegularizationType {
    None,
    L1,      // Lasso: Drives weights to zero (feature selection)
    L2,      // Ridge: Proportional shrinkage (prevents large weights)
    L1L2 {   // Elastic Net: Combination
        l1_ratio: f64,  // 0.0 = pure L2, 1.0 = pure L1
    },
}

pub struct NeuralNetwork {
    // ... existing fields
    regularization: RegularizationType,
    lambda: f64,  // Regularization strength
}

impl NeuralNetwork {
    pub fn new_with_regularization(
        layer_sizes: &[usize],
        activations: &[ActivationType],
        learning_rate: f64,
        regularization: RegularizationType,
        lambda: f64,
    ) -> Self {
        assert!(lambda >= 0.0, "Lambda must be non-negative, got: {}", lambda);
        // ...
    }

    /// Builder pattern for convenient construction
    pub fn builder() -> NeuralNetworkBuilder {
        NeuralNetworkBuilder::new()
    }
}
```

### Mathematical Implementation

**L2 Regularization (Ridge):**
```
Loss = MSE + (lambda / 2N) × Σ(weights²)
Gradient: ∂Loss/∂w = ∂MSE/∂w + (lambda / N) × w
```

**L1 Regularization (Lasso):**
```
Loss = MSE + (lambda / N) × Σ|weights|
Gradient: ∂Loss/∂w = ∂MSE/∂w + (lambda / N) × sign(w)
```

**Implementation:**
```rust
// neural_network/src/lib.rs

impl NeuralNetwork {
    /// Compute regularization loss (added to MSE)
    fn regularization_loss(&self) -> f64 {
        let n = self.layers[0].weights.rows as f64; // Number of samples

        match self.regularization {
            RegularizationType::None => 0.0,

            RegularizationType::L2 => {
                // L2: (lambda / 2N) × Σ(w²)
                let weight_sum_squares: f64 = self.layers.iter()
                    .map(|layer| {
                        layer.weights.data.iter()
                            .map(|w| w * w)
                            .sum::<f64>()
                    })
                    .sum();

                (self.lambda / (2.0 * n)) * weight_sum_squares
            }

            RegularizationType::L1 => {
                // L1: (lambda / N) × Σ|w|
                let weight_sum_abs: f64 = self.layers.iter()
                    .map(|layer| {
                        layer.weights.data.iter()
                            .map(|w| w.abs())
                            .sum::<f64>()
                    })
                    .sum();

                (self.lambda / n) * weight_sum_abs
            }

            RegularizationType::L1L2 { l1_ratio } => {
                // Elastic Net: alpha × L1 + (1-alpha) × L2
                let l1_loss = /* compute L1 as above */;
                let l2_loss = /* compute L2 as above */;
                l1_ratio * l1_loss + (1.0 - l1_ratio) * l2_loss
            }
        }
    }

    /// Apply regularization gradient during backprop
    fn apply_regularization_gradient(&self, layer_idx: usize, weight_gradient: &mut Matrix<f64>) {
        let n = self.layers[0].weights.rows as f64;

        match self.regularization {
            RegularizationType::None => {}

            RegularizationType::L2 => {
                // Add (lambda / N) × w to gradient
                let weights = &self.layers[layer_idx].weights;
                for i in 0..weight_gradient.rows {
                    for j in 0..weight_gradient.cols {
                        weight_gradient[(i, j)] += (self.lambda / n) * weights[(i, j)];
                    }
                }
            }

            RegularizationType::L1 => {
                // Add (lambda / N) × sign(w) to gradient
                let weights = &self.layers[layer_idx].weights;
                for i in 0..weight_gradient.rows {
                    for j in 0..weight_gradient.cols {
                        let sign = if weights[(i, j)] > 0.0 { 1.0 }
                                   else if weights[(i, j)] < 0.0 { -1.0 }
                                   else { 0.0 };
                        weight_gradient[(i, j)] += (self.lambda / n) * sign;
                    }
                }
            }

            // ... L1L2 combination
        }
    }

    /// Backpropagation with regularization
    pub fn backward(&mut self, target: &Matrix<f64>) {
        // 1. Standard backprop (existing code)
        // ... compute gradients

        // 2. Add regularization gradient to each layer
        for (layer_idx, layer_gradient) in self.layer_gradients.iter_mut().enumerate() {
            self.apply_regularization_gradient(layer_idx, &mut layer_gradient.weights);
            // Note: Don't regularize biases!
        }
    }

    /// Total loss including regularization
    pub fn total_loss(&self, predictions: &Matrix<f64>, targets: &Matrix<f64>) -> f64 {
        let mse = self.mse_loss(predictions, targets);
        let reg_loss = self.regularization_loss();
        mse + reg_loss
    }
}
```

## Technical Approach

### Implementation Phases

**Phase 1: Core Implementation (Week 1, Days 1-3)**

Files to modify:
- `neural_network/src/lib.rs` - Add regularization fields, methods

```rust
// 1. Add RegularizationType enum (top of file)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegularizationType {
    None,
    L1,
    L2,
    L1L2 { l1_ratio: f64 },
}

// 2. Modify NeuralNetwork struct
pub struct NeuralNetwork {
    // ... existing fields
    pub regularization: RegularizationType,
    pub lambda: f64,
}

// 3. Update constructor
impl NeuralNetwork {
    pub fn new(
        layer_sizes: &[usize],
        activations: &[ActivationType],
        learning_rate: f64,
    ) -> Self {
        // Defaults to no regularization
        Self::new_with_regularization(
            layer_sizes,
            activations,
            learning_rate,
            RegularizationType::None,
            0.0,
        )
    }

    pub fn new_with_regularization(
        layer_sizes: &[usize],
        activations: &[ActivationType],
        learning_rate: f64,
        regularization: RegularizationType,
        lambda: f64,
    ) -> Self {
        assert!(lambda >= 0.0, "Lambda must be non-negative");
        if let RegularizationType::L1L2 { l1_ratio } = regularization {
            assert!(l1_ratio >= 0.0 && l1_ratio <= 1.0,
                "L1 ratio must be in [0, 1], got: {}", l1_ratio);
        }

        // ... existing construction code

        Self {
            // ... existing fields
            regularization,
            lambda,
        }
    }

    // 4. Add builder pattern
    pub fn builder() -> NeuralNetworkBuilder {
        NeuralNetworkBuilder::new()
    }
}

// 5. Builder implementation
pub struct NeuralNetworkBuilder {
    layer_sizes: Vec<usize>,
    activations: Vec<ActivationType>,
    learning_rate: f64,
    regularization: RegularizationType,
    lambda: f64,
}

impl NeuralNetworkBuilder {
    pub fn new() -> Self {
        Self {
            layer_sizes: Vec::new(),
            activations: Vec::new(),
            learning_rate: 0.01,
            regularization: RegularizationType::None,
            lambda: 0.0,
        }
    }

    pub fn layer_sizes(mut self, sizes: &[usize]) -> Self {
        self.layer_sizes = sizes.to_vec();
        self
    }

    pub fn activations(mut self, acts: &[ActivationType]) -> Self {
        self.activations = acts.to_vec();
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn l1_regularization(mut self, lambda: f64) -> Self {
        self.regularization = RegularizationType::L1;
        self.lambda = lambda;
        self
    }

    pub fn l2_regularization(mut self, lambda: f64) -> Self {
        self.regularization = RegularizationType::L2;
        self.lambda = lambda;
        self
    }

    pub fn build(self) -> NeuralNetwork {
        NeuralNetwork::new_with_regularization(
            &self.layer_sizes,
            &self.activations,
            self.learning_rate,
            self.regularization,
            self.lambda,
        )
    }
}
```

**Phase 2: Testing (Week 1, Days 4-5)**

Create comprehensive test suite in `neural_network/tests/regularization_tests.rs`:

```rust
use neural_network::{NeuralNetwork, RegularizationType, ActivationType};
use linear_algebra::matrix::Matrix;
use approx::assert_relative_eq;

// ===================================================================
// Test 1: L2 Regularization Reduces Weight Magnitude
// ===================================================================

#[test]
fn test_l2_reduces_weights() {
    // Small dataset (prone to overfitting)
    let X = Matrix::from_vec(vec![0.0, 1.0, 2.0, 3.0], 4, 1).unwrap();
    let y = Matrix::from_vec(vec![0.0, 1.0, 2.0, 3.0], 4, 1).unwrap();

    // Without regularization
    let mut nn_no_reg = NeuralNetwork::new(&[1, 10, 1], &[ActivationType::ReLU, ActivationType::Linear], 0.01);
    nn_no_reg.fit(&X, &y, 1000, 0);
    let weights_no_reg: f64 = nn_no_reg.layers.iter()
        .map(|l| l.weights.data.iter().map(|w| w.abs()).sum::<f64>())
        .sum();

    // With L2 regularization
    let mut nn_l2 = NeuralNetwork::builder()
        .layer_sizes(&[1, 10, 1])
        .activations(&[ActivationType::ReLU, ActivationType::Linear])
        .learning_rate(0.01)
        .l2_regularization(0.01)
        .build();
    nn_l2.fit(&X, &y, 1000, 0);
    let weights_l2: f64 = nn_l2.layers.iter()
        .map(|l| l.weights.data.iter().map(|w| w.abs()).sum::<f64>())
        .sum();

    // L2 should produce smaller weights
    assert!(weights_l2 < weights_no_reg,
        "L2 regularization should reduce total weight magnitude");
}

// ===================================================================
// Test 2: L1 Regularization Produces Sparse Weights
// ===================================================================

#[test]
fn test_l1_sparsity() {
    let X = Matrix::from_vec(vec![/* data */], n, features).unwrap();
    let y = Matrix::from_vec(vec![/* labels */], n, 1).unwrap();

    let mut nn = NeuralNetwork::builder()
        .layer_sizes(&[features, 20, 1])
        .activations(&[ActivationType::ReLU, ActivationType::Linear])
        .l1_regularization(0.1)
        .build();

    nn.fit(&X, &y, 1000, 0);

    // Count near-zero weights
    let near_zero_count = nn.layers.iter()
        .map(|l| l.weights.data.iter().filter(|&&w| w.abs() < 0.01).count())
        .sum::<usize>();

    let total_weights: usize = nn.layers.iter()
        .map(|l| l.weights.data.len())
        .sum();

    // L1 should drive significant portion to zero
    let sparsity_ratio = near_zero_count as f64 / total_weights as f64;
    assert!(sparsity_ratio > 0.2, "L1 should produce at least 20% sparse weights");
}

// ===================================================================
// Test 3: Gradient Check with Regularization
// ===================================================================

#[test]
fn test_regularization_gradient_numerical() {
    let mut nn = NeuralNetwork::builder()
        .layer_sizes(&[2, 3, 1])
        .l2_regularization(0.1)
        .build();

    // Small input for gradient checking
    let X = Matrix::from_vec(vec![1.0, 2.0], 1, 2).unwrap();
    let y = Matrix::from_vec(vec![1.5], 1, 1).unwrap();

    // Compute analytical gradient
    let predictions = nn.forward(&X);
    nn.backward(&y);
    let analytical_grad = nn.layers[0].weights.clone();

    // Compute numerical gradient
    let epsilon = 1e-5;
    let mut numerical_grad = Matrix::zeros(
        analytical_grad.rows,
        analytical_grad.cols,
    );

    for i in 0..analytical_grad.rows {
        for j in 0..analytical_grad.cols {
            // w + epsilon
            nn.layers[0].weights[(i, j)] += epsilon;
            let loss_plus = nn.total_loss(&nn.forward(&X), &y);

            // w - epsilon
            nn.layers[0].weights[(i, j)] -= 2.0 * epsilon;
            let loss_minus = nn.total_loss(&nn.forward(&X), &y);

            // Restore
            nn.layers[0].weights[(i, j)] += epsilon;

            // Numerical gradient
            numerical_grad[(i, j)] = (loss_plus - loss_minus) / (2.0 * epsilon);
        }
    }

    // Compare (should be within 1e-5)
    for i in 0..analytical_grad.rows {
        for j in 0..analytical_grad.cols {
            assert_relative_eq!(
                analytical_grad[(i, j)],
                numerical_grad[(i, j)],
                epsilon = 1e-4
            );
        }
    }
}

// ===================================================================
// Test 4: Regularization Improves Generalization
// ===================================================================

#[test]
fn test_regularization_generalization() {
    // Create train/test split
    let (X_train, y_train, X_test, y_test) = create_polynomial_data();

    // Train without regularization
    let mut nn_no_reg = NeuralNetwork::new(&[1, 20, 1], &[ActivationType::Tanh, ActivationType::Linear], 0.01);
    nn_no_reg.fit(&X_train, &y_train, 500, 0);
    let test_loss_no_reg = nn_no_reg.mse_loss(&nn_no_reg.forward(&X_test), &y_test);

    // Train with regularization
    let mut nn_l2 = NeuralNetwork::builder()
        .layer_sizes(&[1, 20, 1])
        .activations(&[ActivationType::Tanh, ActivationType::Linear])
        .l2_regularization(0.01)
        .build();
    nn_l2.fit(&X_train, &y_train, 500, 0);
    let test_loss_l2 = nn_l2.mse_loss(&nn_l2.forward(&X_test), &y_test);

    // L2 should generalize better (lower test loss)
    assert!(test_loss_l2 < test_loss_no_reg,
        "L2 should improve generalization on test set");
}

// Helper function
fn create_polynomial_data() -> (Matrix<f64>, Matrix<f64>, Matrix<f64>, Matrix<f64>) {
    // Generate y = x^2 + noise
    // Return (X_train, y_train, X_test, y_test)
}
```

**Phase 3: Interactive Demo (Week 2)**

Create visualization in `web/src/components/regularization_demo.rs`:

```rust
//! Interactive Regularization Demonstration
//!
//! Shows overfitting vs regularization with real-time training

use dioxus::prelude::*;
use neural_network::{NeuralNetwork, RegularizationType, ActivationType};
use linear_algebra::matrix::Matrix;

#[component]
pub fn RegularizationDemo() -> Element {
    // State
    let mut lambda = use_signal(|| 0.0);
    let mut reg_type = use_signal(|| RegularizationType::L2);
    let mut is_training = use_signal(|| false);
    let mut epoch = use_signal(|| 0);

    // Model states
    let mut model_no_reg = use_signal(|| create_model(RegularizationType::None, 0.0));
    let mut model_with_reg = use_signal(|| create_model(*reg_type.read(), *lambda.read()));

    // Training data (small dataset to induce overfitting)
    let train_data = use_signal(|| generate_noisy_polynomial(n_points: 20));
    let test_data = use_signal(|| generate_noisy_polynomial(n_points: 100));

    // Training loop
    let train_step = move |_| {
        if *is_training.read() {
            // Train both models for 10 iterations
            for _ in 0..10 {
                model_no_reg.write().fit(&train_data.X, &train_data.y, 1, 0);
                model_with_reg.write().fit(&train_data.X, &train_data.y, 1, 0);
            }
            epoch.set(*epoch.read() + 10);
        }
    };

    use_effect(move || {
        if *is_training.read() {
            request_animation_frame(train_step);
        }
    });

    rsx! {
        div { class: "regularization-demo",
            // Controls
            div { class: "controls",
                h2 { "Regularization Demo" }

                // Regularization type selector
                select {
                    value: "{reg_type:?}",
                    onchange: move |e| {
                        let new_type = match e.value.as_str() {
                            "L1" => RegularizationType::L1,
                            "L2" => RegularizationType::L2,
                            _ => RegularizationType::None,
                        };
                        reg_type.set(new_type);
                        model_with_reg.set(create_model(new_type, *lambda.read()));
                    },
                    option { value: "L2", "L2 (Ridge)" }
                    option { value: "L1", "L1 (Lasso)" }
                }

                // Lambda slider
                label { "Regularization Strength (λ): {lambda:.4}" }
                input {
                    r#type: "range",
                    min: "0.0",
                    max: "0.1",
                    step: "0.001",
                    value: "{lambda}",
                    oninput: move |e| {
                        let new_lambda: f64 = e.value.parse().unwrap();
                        lambda.set(new_lambda);
                        model_with_reg.set(create_model(*reg_type.read(), new_lambda));
                    }
                }

                // Train/Pause button
                button {
                    onclick: move |_| is_training.set(!*is_training.read()),
                    if *is_training.read() { "Pause" } else { "Train" }
                }

                // Reset button
                button {
                    onclick: move |_| {
                        epoch.set(0);
                        model_no_reg.set(create_model(RegularizationType::None, 0.0));
                        model_with_reg.set(create_model(*reg_type.read(), *lambda.read()));
                    },
                    "Reset"
                }

                p { "Epoch: {epoch}" }
            }

            // Visualization
            div { class: "visualization",
                // Scatter plot with model predictions
                svg {
                    width: "800",
                    height: "400",

                    // Training data points
                    for (x, y) in train_data.points() {
                        circle {
                            cx: "{scale_x(x)}",
                            cy: "{scale_y(y)}",
                            r: "4",
                            fill: "blue",
                        }
                    }

                    // Model predictions (no regularization) - Red line
                    polyline {
                        points: "{generate_prediction_line(&model_no_reg.read(), &test_data.X)}",
                        stroke: "red",
                        stroke_width: "2",
                        fill: "none",
                    }

                    // Model predictions (with regularization) - Green line
                    polyline {
                        points: "{generate_prediction_line(&model_with_reg.read(), &test_data.X)}",
                        stroke: "green",
                        stroke_width: "2",
                        fill: "none",
                    }

                    // True function (black line)
                    polyline {
                        points: "{generate_true_function_line()}",
                        stroke: "black",
                        stroke_width: "1",
                        stroke_dasharray: "5,5",
                        fill: "none",
                    }
                }

                // Legend
                div { class: "legend",
                    div { "● Blue: Training data" }
                    div { "— Red: No regularization" }
                    div { "— Green: With regularization (λ={lambda:.3})" }
                    div { "- - Black: True function" }
                }
            }

            // Loss comparison
            div { class: "loss-chart",
                h3 { "Loss Comparison" }
                LossChart {
                    train_loss_no_reg: compute_loss(&model_no_reg.read(), &train_data),
                    test_loss_no_reg: compute_loss(&model_no_reg.read(), &test_data),
                    train_loss_reg: compute_loss(&model_with_reg.read(), &train_data),
                    test_loss_reg: compute_loss(&model_with_reg.read(), &test_data),
                }
            }

            // Weight magnitude comparison
            div { class: "weight-viz",
                h3 { "Weight Magnitudes" }
                WeightHistogram {
                    weights_no_reg: extract_weights(&model_no_reg.read()),
                    weights_reg: extract_weights(&model_with_reg.read()),
                }
            }
        }
    }
}

fn create_model(reg_type: RegularizationType, lambda: f64) -> NeuralNetwork {
    NeuralNetwork::builder()
        .layer_sizes(&[1, 20, 1])  // Overparameterized to demonstrate overfitting
        .activations(&[ActivationType::Tanh, ActivationType::Linear])
        .learning_rate(0.01)
        .regularization(reg_type)
        .lambda(lambda)
        .build()
}

fn generate_noisy_polynomial(n_points: usize) -> DataSet {
    // Generate y = x^2 + noise
    // Returns DataSet { X, y, points }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] `RegularizationType` enum supports None, L1, L2, L1L2
- [ ] `NeuralNetwork::new()` maintains backward compatibility (defaults to no regularization)
- [ ] `NeuralNetwork::builder()` provides fluent API for regularization
- [ ] L2 regularization correctly adds `(lambda/2N) × Σw²` to loss
- [ ] L1 regularization correctly adds `(lambda/N) × Σ|w|` to loss
- [ ] Regularization gradient correctly applied during backpropagation
- [ ] **Biases are NOT regularized** (only weights)
- [ ] Lambda validation (must be >= 0)
- [ ] L1L2 ratio validation (must be in [0, 1])

### Non-Functional Requirements
- [ ] **Performance:** No measurable slowdown (<5% overhead)
- [ ] **Memory:** No additional allocations in hot path
- [ ] **Backward compatibility:** Existing code works without changes
- [ ] **Documentation:** All public APIs have rustdoc comments

### Quality Gates
- [ ] Gradient checking tests pass (analytical vs numerical < 1e-5)
- [ ] L2 reduces weight magnitude on overfitting dataset
- [ ] L1 produces sparse weights (>20% near zero)
- [ ] Regularization improves test set performance
- [ ] All existing tests still pass
- [ ] Interactive demo shows clear overfitting vs regularization

## Success Metrics

**Technical:**
- Gradient checking error < 1e-5
- L2 reduces weight magnitude by 30-70% on overfitting test
- L1 produces 20-50% sparse weights
- Implementation adds <100 lines of code

**Educational:**
- Users can explain regularization after 5 min with demo
- 80% understand lambda controls strength
- 70% understand L1 vs L2 difference

## Dependencies & Prerequisites

**No new dependencies** - uses existing `linear_algebra` and `neural_network` code.

**Prerequisites:**
- [ ] Understand L1/L2 mathematics
- [ ] Review gradient descent update rule
- [ ] Study overfitting phenomenon

## Resources

**Academic Papers:**
- L2 (Ridge): Hoerl & Kennard (1970) - https://www.jstor.org/stable/1267351
- L1 (Lasso): Tibshirani (1996) - https://www.jstor.org/stable/2346178
- Elastic Net: Zou & Hastie (2005) - https://www.jstor.org/stable/3647580

**Tutorials:**
- CS229 Regularization: https://cs229.stanford.edu/notes2021fall/cs229-notes1.pdf
- Regularization explained: https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a

**Internal:**
- `neural_network/src/lib.rs:179-245` - Backpropagation implementation
- `docs/TECHNICAL_BOOK.md` - Chapter on neural networks

---

**Estimated Effort:** 40 hours (1-2 weeks)
**Target Release:** v0.2.5 (before CNN)
**Priority:** 🟡 HIGH - Quick win with high educational value

**Files to modify:**
- `neural_network/src/lib.rs` - Add regularization logic
- `neural_network/tests/regularization_tests.rs` - New test file
- `web/src/components/regularization_demo.rs` - New visualization
