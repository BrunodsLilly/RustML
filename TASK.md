# Interactive Gradient Descent Optimizer Visualizer 🎯

## Overview

Build an **Interactive Gradient Descent Optimizer Comparison Platform** that visually demonstrates how modern optimizers (SGD, Momentum, RMSprop, Adam) navigate loss landscapes differently. This feature combines:
- **Educational depth**: Addresses the #1 practical challenge in neural network training
- **Visual impact**: Animated optimizer paths on 2D/3D loss surfaces
- **Industry relevance**: Adam/RMSprop used in 95%+ of production ML
- **Implementation feasibility**: Builds directly on existing gradient descent code

## Implementation Plan

### Phase 1: Core Optimizer Library (Week 1-2)
**Location**: `neural_network/src/optimizer.rs`
- Implement `Optimizer` enum (SGD, Momentum, RMSprop, Adam)
- Add optimizer state management (velocity, squared gradients, timestep)
- Integrate into existing `NeuralNetwork::fit()` and `LinearRegressor::fit()`
- Unit tests comparing convergence rates on standard benchmarks
- Example: Rosenbrock function optimization demo

### Phase 2: Web Integration (Week 2-3)
**Location**: `web/src/components/optimizer_demo.rs`
- Side-by-side optimizer comparison UI (4 parallel trainings)
- Interactive hyperparameter controls (β₁, β₂, learning rate)
- Real-time convergence metrics display
- Integration with existing gradient descent trainer

### Phase 3: Advanced Visualizations (Week 3-4)
**Enhancements to web component:**
- 2D loss landscape heatmap with optimizer paths as colored traces
- Animated "ball rolling" visualization showing momentum effects
- Convergence comparison charts (iterations-to-convergence)
- Educational annotations explaining optimizer behavior at key moments

### Phase 4: Educational Content & Polish (Week 4)
- Interactive tutorials explaining each optimizer
- Common failure modes and debugging tips
- Export functionality for optimizer paths
- Documentation and examples

## Technical Approach

### Rust Library Changes
```rust
// neural_network/src/optimizer.rs
pub enum OptimizerType { SGD, Momentum, RMSprop, Adam }

pub struct Optimizer {
    optimizer_type: OptimizerType,
    learning_rate: f64,
    velocity: Option<Matrix<f64>>,
    squared_gradients: Option<Matrix<f64>>,
    beta1: f64,  // 0.9
    beta2: f64,  // 0.999
    epsilon: f64, // 1e-8
    timestep: usize,
}
```

### Web Component Architecture
- Dioxus signals for reactive state
- Async training with UI updates every N iterations
- SVG-based loss landscape visualization
- CSS animations for smooth transitions
- Reuse patterns from existing `GradientDescentDemo`

## Success Metrics
- Users can compare 4 optimizers simultaneously on same dataset
- Visual clarity: Optimizer paths clearly distinguishable on loss surface
- Educational impact: Users understand when to use each optimizer
- Performance: 1000 iterations train in <2 seconds on WASM

## Why This Feature?

### Educational Value ⭐⭐⭐⭐⭐
- Addresses THE most common practical bottleneck in NN training
- Highly visual - optimizer behavior is immediately intuitive
- Bridges theory (math) with practice (when to use what)

### Industry Relevance ⭐⭐⭐⭐⭐
- Adam is default in TensorFlow, PyTorch, research papers
- Understanding optimizer behavior critical for debugging training

### Implementation Feasibility ⭐⭐⭐⭐⭐
- Builds directly on existing gradient descent code
- No new dependencies (pure Rust math)
- WASM-friendly (all calculations client-side)
- Reuses visualization patterns from showcase

### Alternative Considered
**CNNs**: Higher educational value but:
- Much higher complexity (2D convolutions, pooling)
- Memory-intensive (benefits from GPU, limited in WASM)
- Best after mastering optimizers first

**Regularization (L1/L2, Dropout)**: Also excellent but:
- Less visual impact than animated optimizer paths
- Good follow-up feature after optimizers

## Estimated Effort
- **Total**: 70-100 hours over 4 weeks
- **Phase 1** (Library): 25-30 hours
- **Phase 2** (Web Integration): 20-25 hours
- **Phase 3** (Visualizations): 15-20 hours
- **Phase 4** (Polish): 10-15 hours

## Dependencies
- Existing: `linear_algebra`, `neural_network`, `linear_regression`
- New web dependencies: `dioxus-charts` (optional), `dioxus-motion` (animations)
- No breaking changes to existing APIs

## Future Extensions
- Multi-class classification with Softmax + Cross-Entropy
- Regularization techniques (L1/L2, Dropout)
- Batch normalization
- Convolutional neural networks

---

# Detailed Implementation Specification

## Problem Statement / Motivation

### Current State
- ✅ Basic neural network with backpropagation (23 tests passing)
- ✅ XOR problem solved with fixed learning rate SGD
- ✅ Gradient descent visualizer for linear regression
- ❌ **No optimizer options** - users stuck with vanilla gradient descent
- ❌ No visual explanation of why training sometimes fails to converge
- ❌ No intuition for learning rate sensitivity

### The Problem
**Vanilla gradient descent has critical limitations:**
1. **Learning rate sensitivity**: Too high → divergence, too low → painfully slow
2. **Ravines**: Oscillates instead of following the valley
3. **Saddle points**: Gets stuck in flat regions
4. **Local minima**: Limited escaping capability

**Modern optimizers solve these**, but learners don't understand *how* or *when* to use them.

### Educational Gap
Existing resources either:
- Show optimizer equations without intuition (academic papers)
- Use pre-trained models without showing the training process (TensorFlow Playground)
- Treat optimizers as magic black boxes (most tutorials)

**This feature bridges that gap** with interactive, real-time visualization.

## Proposed Solution

### High-Level Approach

Build a **four-panel comparison interface** where users can:
1. Select a loss function (Rosenbrock, Beale, saddle point, or custom)
2. Configure 4 optimizers simultaneously with different hyperparameters
3. Click "Train" and watch animated optimizer paths converge on a 2D loss landscape
4. Compare convergence speed, path smoothness, and final accuracy

### Core Components

#### 1. Rust Optimizer Library (`neural_network/src/optimizer.rs`)
```rust
pub enum OptimizerType {
    SGD,           // Vanilla gradient descent
    Momentum,      // Velocity-based acceleration
    RMSprop,       // Adaptive per-parameter learning rates
    Adam,          // Combines Momentum + RMSprop (industry standard)
}

pub struct Optimizer {
    optimizer_type: OptimizerType,
    learning_rate: f64,

    // Momentum state
    velocity_weights: Option<Vec<Matrix<f64>>>,
    velocity_bias: Option<Vec<Vector<f64>>>,
    beta1: f64,  // Momentum decay (typical: 0.9)

    // RMSprop/Adam state
    squared_gradients_weights: Option<Vec<Matrix<f64>>>,
    squared_gradients_bias: Option<Vec<Vector<f64>>>,
    beta2: f64,  // RMS decay (typical: 0.999)
    epsilon: f64, // Numerical stability (1e-8)

    // Adam-specific
    timestep: usize,
}

impl Optimizer {
    pub fn new(optimizer_type: OptimizerType, learning_rate: f64) -> Self;

    pub fn update_weights(
        &mut self,
        layer_idx: usize,
        weight_gradients: &Matrix<f64>,
        current_weights: &mut Matrix<f64>
    );

    pub fn update_bias(
        &mut self,
        layer_idx: usize,
        bias_gradients: &Vector<f64>,
        current_bias: &mut Vector<f64>
    );

    pub fn reset(&mut self);  // Reset state for new training run
}
```

#### 2. Integration Points

**`neural_network/src/lib.rs`** modifications:
```rust
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub optimizer: Optimizer,  // NEW: Replace learning_rate field
    pub history: TrainingHistory,
}

impl NeuralNetwork {
    pub fn new(
        layer_sizes: &[usize],
        activations: &[ActivationType],
        optimizer: Optimizer,  // NEW: Accept optimizer config
    ) -> Self;

    // backward() now delegates weight updates to optimizer
    fn backward(&mut self, input: &Vector<f64>, target: &Vector<f64>) -> f64 {
        // ... compute gradients ...

        // Update weights using optimizer (not raw gradient descent)
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            self.optimizer.update_weights(
                layer_idx,
                &weight_gradients[layer_idx],
                &mut layer.weights
            );
            self.optimizer.update_bias(
                layer_idx,
                &bias_gradients[layer_idx],
                &mut layer.biases
            );
        }

        loss
    }
}
```

**`linear_regression/src/lib.rs`** modifications:
```rust
pub struct LinearRegressor {
    pub weights: Matrix<f64>,
    pub bias: f64,
    pub optimizer: Optimizer,  // NEW
    pub training_history: Vec<f64>,
}
```

#### 3. Web Component (`web/src/components/optimizer_demo.rs`)

```rust
#[component]
pub fn OptimizerDemo() -> Element {
    // State management
    let mut optimizers = use_signal(|| vec![
        Optimizer::new(OptimizerType::SGD, 0.01),
        Optimizer::new(OptimizerType::Momentum, 0.01),
        Optimizer::new(OptimizerType::RMSprop, 0.01),
        Optimizer::new(OptimizerType::Adam, 0.01),
    ]);

    let mut loss_function = use_signal(|| LossFunction::Rosenbrock);
    let mut is_training = use_signal(|| false);
    let mut optimizer_paths = use_signal(|| vec![vec![]; 4]);
    let mut current_iteration = use_signal(|| 0);

    rsx! {
        div { class: "optimizer-demo",
            // Top: Loss function selector
            LossFunctionSelector {
                selected: loss_function,
                options: vec!["Rosenbrock", "Beale", "Saddle Point", "Custom"]
            }

            // Middle: Optimizer configuration panels (4 columns)
            div { class: "optimizer-configs",
                for (idx, optimizer) in optimizers().iter().enumerate() {
                    OptimizerConfig {
                        optimizer_type: optimizer.optimizer_type,
                        learning_rate: optimizer.learning_rate,
                        beta1: optimizer.beta1,
                        beta2: optimizer.beta2,
                        on_change: move |new_config| {
                            optimizers.write()[idx] = new_config;
                        }
                    }
                }
            }

            // Train button
            button {
                class: "train-button",
                disabled: is_training(),
                onclick: train_all_optimizers,
                if is_training() { "Training..." } else { "Train All" }
            }

            // Main visualization: 2D loss landscape with paths
            LossLandscapeViz {
                loss_function: loss_function(),
                optimizer_paths: optimizer_paths(),
                current_iteration: current_iteration(),
                animating: is_training()
            }

            // Bottom: Convergence metrics
            ConvergenceMetrics {
                optimizers: optimizers(),
                paths: optimizer_paths(),
                final_losses: calculate_final_losses()
            }
        }
    }
}
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│  ┌─────────┬─────────┬─────────┬─────────┐             │
│  │   SGD   │Momentum │ RMSprop │  Adam   │ ← Configs   │
│  └─────────┴─────────┴─────────┴─────────┘             │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              2D Loss Landscape Heatmap                   │
│   ┌───────────────────────────────────────────┐         │
│   │  🔴 SGD path       (red)                  │         │
│   │  🟢 Momentum path  (green)                │         │
│   │  🔵 RMSprop path   (blue)                 │         │
│   │  🟡 Adam path      (yellow)               │         │
│   │                                           │         │
│   │      ⭐ Optimal point                     │         │
│   └───────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│             Convergence Comparison Table                 │
│  Optimizer │ Iterations │ Final Loss │ Path Length      │
│  ──────────┼────────────┼────────────┼──────────        │
│  SGD       │    1000    │   0.234    │   124.5          │
│  Momentum  │     450    │   0.012    │    67.2          │
│  RMSprop   │     380    │   0.008    │    52.1          │
│  Adam      │     320    │   0.003    │    45.8   ⭐     │
└─────────────────────────────────────────────────────────┘
```

## Implementation Phases

### **Phase 1: Core Optimizer Library** (Week 1-2, 25-30 hours)

**Files to create:**
- `neural_network/src/optimizer.rs` (250-300 lines)
- `neural_network/examples/optimizer_comparison.rs` (150 lines)
- `neural_network/tests/optimizer_tests.rs` (200 lines)

**Tasks:**
1. Implement `Optimizer` struct with 4 optimizer types
2. Add state management (velocity, squared gradients, timestep)
3. Implement update rules:
   - **SGD**: `θ = θ - α∇L`
   - **Momentum**: `v = βv + ∇L; θ = θ - αv`
   - **RMSprop**: `s = βs + (1-β)(∇L)²; θ = θ - α∇L/√(s+ε)`
   - **Adam**: Combines momentum + RMSprop with bias correction
4. Unit tests:
   - Convergence on Rosenbrock function
   - Escaping saddle points
   - Learning rate sensitivity comparison
5. Integration into `NeuralNetwork` and `LinearRegressor`

**Success Criteria:**
- Adam converges 2-3x faster than SGD on Rosenbrock
- All tests pass
- Zero breaking changes to existing examples

**Detailed Task Breakdown:**

#### Task 1.1: Create optimizer.rs skeleton
```rust
// neural_network/src/optimizer.rs

use linear_algebra::{matrix::Matrix, vectors::Vector};
use rand::distributions::{Distribution, Uniform};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizerType {
    SGD,
    Momentum,
    RMSprop,
    Adam,
}

#[derive(Debug, Clone)]
pub struct Optimizer {
    optimizer_type: OptimizerType,
    learning_rate: f64,

    // Momentum state
    velocity_weights: Vec<Matrix<f64>>,
    velocity_bias: Vec<Vector<f64>>,
    beta1: f64,

    // RMSprop/Adam state
    squared_gradients_weights: Vec<Matrix<f64>>,
    squared_gradients_bias: Vec<Vector<f64>>,
    beta2: f64,
    epsilon: f64,

    // Adam timestep
    timestep: usize,
}
```

#### Task 1.2: Implement SGD
```rust
impl Optimizer {
    pub fn sgd(learning_rate: f64) -> Self {
        Optimizer {
            optimizer_type: OptimizerType::SGD,
            learning_rate,
            velocity_weights: Vec::new(),
            velocity_bias: Vec::new(),
            beta1: 0.0,
            squared_gradients_weights: Vec::new(),
            squared_gradients_bias: Vec::new(),
            beta2: 0.0,
            epsilon: 0.0,
            timestep: 0,
        }
    }

    pub fn update_weights(
        &mut self,
        layer_idx: usize,
        gradient: &Matrix<f64>,
        weights: &mut Matrix<f64>,
    ) {
        match self.optimizer_type {
            OptimizerType::SGD => {
                // θ = θ - α∇L
                for i in 0..weights.rows {
                    for j in 0..weights.cols {
                        weights[(i, j)] -= self.learning_rate * gradient[(i, j)];
                    }
                }
            },
            _ => unimplemented!()
        }
    }
}
```

#### Task 1.3: Implement Momentum
```rust
pub fn momentum(learning_rate: f64, beta1: f64) -> Self {
    Optimizer {
        optimizer_type: OptimizerType::Momentum,
        learning_rate,
        beta1,
        // ... rest initialized
    }
}

// In update_weights():
OptimizerType::Momentum => {
    // Initialize velocity if needed
    if self.velocity_weights.is_empty() {
        self.initialize_state(num_layers, layer_shapes);
    }

    let v = &mut self.velocity_weights[layer_idx];

    // v = β₁·v + ∇L
    for i in 0..weights.rows {
        for j in 0..weights.cols {
            v[(i, j)] = self.beta1 * v[(i, j)] + gradient[(i, j)];
            weights[(i, j)] -= self.learning_rate * v[(i, j)];
        }
    }
}
```

#### Task 1.4: Implement RMSprop
```rust
pub fn rmsprop(learning_rate: f64, beta2: f64, epsilon: f64) -> Self {
    Optimizer {
        optimizer_type: OptimizerType::RMSprop,
        learning_rate,
        beta2,
        epsilon,
        // ... rest initialized
    }
}

// In update_weights():
OptimizerType::RMSprop => {
    if self.squared_gradients_weights.is_empty() {
        self.initialize_state(num_layers, layer_shapes);
    }

    let s = &mut self.squared_gradients_weights[layer_idx];

    // s = β₂·s + (1-β₂)·(∇L)²
    // θ = θ - α·∇L/√(s+ε)
    for i in 0..weights.rows {
        for j in 0..weights.cols {
            s[(i, j)] = self.beta2 * s[(i, j)] + (1.0 - self.beta2) * gradient[(i, j)].powi(2);
            weights[(i, j)] -= self.learning_rate * gradient[(i, j)] / (s[(i, j)] + self.epsilon).sqrt();
        }
    }
}
```

#### Task 1.5: Implement Adam
```rust
pub fn adam(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
    Optimizer {
        optimizer_type: OptimizerType::Adam,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        timestep: 0,
        // ... rest initialized
    }
}

// In update_weights():
OptimizerType::Adam => {
    if self.velocity_weights.is_empty() {
        self.initialize_state(num_layers, layer_shapes);
    }

    self.timestep += 1;
    let t = self.timestep as f64;

    let m = &mut self.velocity_weights[layer_idx];
    let v = &mut self.squared_gradients_weights[layer_idx];

    for i in 0..weights.rows {
        for j in 0..weights.cols {
            // m = β₁·m + (1-β₁)·∇L
            m[(i, j)] = self.beta1 * m[(i, j)] + (1.0 - self.beta1) * gradient[(i, j)];

            // v = β₂·v + (1-β₂)·(∇L)²
            v[(i, j)] = self.beta2 * v[(i, j)] + (1.0 - self.beta2) * gradient[(i, j)].powi(2);

            // Bias correction
            let m_hat = m[(i, j)] / (1.0 - self.beta1.powf(t));
            let v_hat = v[(i, j)] / (1.0 - self.beta2.powf(t));

            // Update
            weights[(i, j)] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}
```

#### Task 1.6: Write unit tests
```rust
// neural_network/tests/optimizer_tests.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_convergence() {
        // Test on simple quadratic: f(x) = x²
        // Should converge to x = 0
    }

    #[test]
    fn test_momentum_accelerates_convergence() {
        // Momentum should converge faster than SGD
    }

    #[test]
    fn test_adam_handles_ravines() {
        // Adam should handle ravines better than SGD
    }

    #[test]
    fn test_rosenbrock_convergence() {
        // f(x,y) = (1-x)² + 100(y-x²)²
        // Classic test function with narrow curved valley
    }
}
```

#### Task 1.7: Create Rosenbrock example
```rust
// neural_network/examples/optimizer_comparison.rs

use neural_network::optimizer::{Optimizer, OptimizerType};

fn rosenbrock(x: f64, y: f64) -> f64 {
    (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2)
}

fn rosenbrock_gradient(x: f64, y: f64) -> (f64, f64) {
    let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x.powi(2));
    let dy = 200.0 * (y - x.powi(2));
    (dx, dy)
}

fn main() {
    let optimizers = vec![
        ("SGD", Optimizer::sgd(0.001)),
        ("Momentum", Optimizer::momentum(0.001, 0.9)),
        ("RMSprop", Optimizer::rmsprop(0.001, 0.999, 1e-8)),
        ("Adam", Optimizer::adam(0.001, 0.9, 0.999, 1e-8)),
    ];

    for (name, mut optimizer) in optimizers {
        let mut x = -1.0;
        let mut y = 1.0;

        println!("\n{} Optimizer:", name);
        println!("Initial: x={:.4}, y={:.4}, loss={:.4}", x, y, rosenbrock(x, y));

        for iter in 0..1000 {
            let (dx, dy) = rosenbrock_gradient(x, y);

            // Update (simplified - in real impl use matrices)
            optimizer.step(&dx, &dy, &mut x, &mut y);

            if iter % 100 == 0 {
                println!("Iter {}: x={:.4}, y={:.4}, loss={:.4}",
                         iter, x, y, rosenbrock(x, y));
            }
        }

        println!("Final: x={:.4}, y={:.4}, loss={:.4}", x, y, rosenbrock(x, y));
    }
}
```

### **Phase 2: Web Integration** (Week 2-3, 20-25 hours)

**Files to create:**
- `web/src/components/optimizer_demo.rs` (600-800 lines)
- `web/src/components/loss_functions.rs` (200 lines)
- `web/src/components/optimizer_config.rs` (150 lines)

**Tasks:**
1. Create `OptimizerDemo` component with 4-panel layout
2. Implement async training with progress updates
3. Add optimizer configuration UI (sliders for β₁, β₂, learning rate)
4. State management for paths, iterations, convergence
5. Integration with existing navigation

**Success Criteria:**
- 4 optimizers train simultaneously
- UI remains responsive during training
- Clear visual feedback on training progress

**Detailed Task Breakdown:**

#### Task 2.1: Create loss_functions.rs
```rust
// web/src/components/loss_functions.rs

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossFunction {
    Rosenbrock,
    Beale,
    SaddlePoint,
    Custom,
}

impl LossFunction {
    pub fn evaluate(&self, x: f64, y: f64) -> f64 {
        match self {
            LossFunction::Rosenbrock => {
                (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2)
            },
            LossFunction::Beale => {
                (1.5 - x + x*y).powi(2) +
                (2.25 - x + x*y.powi(2)).powi(2) +
                (2.625 - x + x*y.powi(3)).powi(2)
            },
            LossFunction::SaddlePoint => {
                x.powi(2) - y.powi(2)
            },
            LossFunction::Custom => {
                // User-defined
                0.0
            }
        }
    }

    pub fn gradient(&self, x: f64, y: f64) -> (f64, f64) {
        match self {
            LossFunction::Rosenbrock => {
                let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x.powi(2));
                let dy = 200.0 * (y - x.powi(2));
                (dx, dy)
            },
            // ... other gradients
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            LossFunction::Rosenbrock => "Rosenbrock",
            LossFunction::Beale => "Beale",
            LossFunction::SaddlePoint => "Saddle Point",
            LossFunction::Custom => "Custom",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            LossFunction::Rosenbrock => "Classic test function with narrow curved valley",
            LossFunction::Beale => "Multiple local minima",
            LossFunction::SaddlePoint => "Saddle point at origin",
            LossFunction::Custom => "User-defined function",
        }
    }
}
```

#### Task 2.2: Create optimizer_config.rs
```rust
// web/src/components/optimizer_config.rs

use dioxus::prelude::*;
use neural_network::optimizer::{Optimizer, OptimizerType};

#[component]
pub fn OptimizerConfig(
    optimizer_type: OptimizerType,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    on_change: EventHandler<Optimizer>,
) -> Element {
    let mut lr = use_signal(|| learning_rate);
    let mut b1 = use_signal(|| beta1);
    let mut b2 = use_signal(|| beta2);

    let update_optimizer = move |_| {
        let new_optimizer = match optimizer_type {
            OptimizerType::SGD => Optimizer::sgd(lr()),
            OptimizerType::Momentum => Optimizer::momentum(lr(), b1()),
            OptimizerType::RMSprop => Optimizer::rmsprop(lr(), b2(), 1e-8),
            OptimizerType::Adam => Optimizer::adam(lr(), b1(), b2(), 1e-8),
        };
        on_change.call(new_optimizer);
    };

    rsx! {
        div { class: "optimizer-config",
            h3 { "{optimizer_type:?}" }

            // Learning rate slider
            div { class: "param-control",
                label { "Learning Rate: {lr():.4}" }
                input {
                    r#type: "range",
                    min: "0.0001",
                    max: "0.1",
                    step: "0.0001",
                    value: "{lr()}",
                    oninput: move |e| {
                        if let Ok(val) = e.value().parse::<f64>() {
                            lr.set(val);
                            update_optimizer(e);
                        }
                    }
                }
            }

            // Beta1 slider (if applicable)
            if matches!(optimizer_type, OptimizerType::Momentum | OptimizerType::Adam) {
                div { class: "param-control",
                    label { "β₁ (Momentum): {b1():.3}" }
                    input {
                        r#type: "range",
                        min: "0.0",
                        max: "0.999",
                        step: "0.001",
                        value: "{b1()}",
                        oninput: move |e| {
                            if let Ok(val) = e.value().parse::<f64>() {
                                b1.set(val);
                                update_optimizer(e);
                            }
                        }
                    }
                }
            }

            // Beta2 slider (if applicable)
            if matches!(optimizer_type, OptimizerType::RMSprop | OptimizerType::Adam) {
                div { class: "param-control",
                    label { "β₂ (RMS): {b2():.4}" }
                    input {
                        r#type: "range",
                        min: "0.9",
                        max: "0.9999",
                        step: "0.0001",
                        value: "{b2()}",
                        oninput: move |e| {
                            if let Ok(val) = e.value().parse::<f64>() {
                                b2.set(val);
                                update_optimizer(e);
                            }
                        }
                    }
                }
            }

            // Preset buttons
            div { class: "presets",
                button {
                    onclick: move |_| {
                        lr.set(0.001);
                        b1.set(0.9);
                        b2.set(0.999);
                        update_optimizer(_);
                    },
                    "Default"
                }
                button {
                    onclick: move |_| {
                        lr.set(0.01);
                        b1.set(0.9);
                        b2.set(0.999);
                        update_optimizer(_);
                    },
                    "Fast"
                }
                button {
                    onclick: move |_| {
                        lr.set(0.0001);
                        b1.set(0.99);
                        b2.set(0.9999);
                        update_optimizer(_);
                    },
                    "Stable"
                }
            }
        }
    }
}
```

#### Task 2.3: Create main OptimizerDemo component
```rust
// web/src/components/optimizer_demo.rs

use dioxus::prelude::*;
use neural_network::optimizer::{Optimizer, OptimizerType};

#[derive(Clone, Debug)]
struct OptimizerPath {
    points: Vec<(f64, f64)>,
    losses: Vec<f64>,
}

#[component]
pub fn OptimizerDemo() -> Element {
    // State
    let mut optimizers = use_signal(|| vec![
        Optimizer::sgd(0.001),
        Optimizer::momentum(0.001, 0.9),
        Optimizer::rmsprop(0.001, 0.999, 1e-8),
        Optimizer::adam(0.001, 0.9, 0.999, 1e-8),
    ]);

    let mut loss_function = use_signal(|| LossFunction::Rosenbrock);
    let mut is_training = use_signal(|| false);
    let mut optimizer_paths = use_signal(|| vec![
        OptimizerPath { points: vec![(-1.0, 1.0)], losses: vec![] },
        OptimizerPath { points: vec![(-1.0, 1.0)], losses: vec![] },
        OptimizerPath { points: vec![(-1.0, 1.0)], losses: vec![] },
        OptimizerPath { points: vec![(-1.0, 1.0)], losses: vec![] },
    ]);
    let mut current_iteration = use_signal(|| 0);
    let max_iterations = use_signal(|| 1000);

    // Training function
    let train_all = move |_| {
        spawn(async move {
            is_training.set(true);
            current_iteration.set(0);

            // Reset paths
            optimizer_paths.write().iter_mut().for_each(|path| {
                path.points = vec![(-1.0, 1.0)];
                path.losses = vec![];
            });

            for iter in 0..max_iterations() {
                // Update each optimizer
                for (idx, optimizer) in optimizers.write().iter_mut().enumerate() {
                    let path = &mut optimizer_paths.write()[idx];
                    let (x, y) = *path.points.last().unwrap();

                    // Compute gradient
                    let (dx, dy) = loss_function().gradient(x, y);

                    // Update position (simplified - actual impl uses matrices)
                    let (new_x, new_y) = optimizer.step(x, y, dx, dy);

                    // Record
                    path.points.push((new_x, new_y));
                    path.losses.push(loss_function().evaluate(new_x, new_y));
                }

                // Update UI every 10 iterations
                if iter % 10 == 0 {
                    current_iteration.set(iter);
                    TimeoutFuture::from_ms(0).await;
                }
            }

            current_iteration.set(max_iterations());
            is_training.set(false);
        });
    };

    rsx! {
        div { id: "optimizer-demo",
            h1 { "Optimizer Comparison Visualizer" }

            // Loss function selector
            div { class: "loss-selector",
                label { "Loss Function:" }
                select {
                    value: "{loss_function():?}",
                    onchange: move |e| {
                        // Parse and update
                    },
                    option { value: "Rosenbrock", "Rosenbrock (Curved Valley)" }
                    option { value: "Beale", "Beale (Multiple Minima)" }
                    option { value: "SaddlePoint", "Saddle Point" }
                }
                p { class: "description",
                    "{loss_function().description()}"
                }
            }

            // Optimizer configs (4 columns)
            div { class: "optimizer-configs-grid",
                OptimizerConfig {
                    optimizer_type: OptimizerType::SGD,
                    learning_rate: 0.001,
                    beta1: 0.0,
                    beta2: 0.0,
                    on_change: move |opt| {
                        optimizers.write()[0] = opt;
                    }
                }
                OptimizerConfig {
                    optimizer_type: OptimizerType::Momentum,
                    learning_rate: 0.001,
                    beta1: 0.9,
                    beta2: 0.0,
                    on_change: move |opt| {
                        optimizers.write()[1] = opt;
                    }
                }
                OptimizerConfig {
                    optimizer_type: OptimizerType::RMSprop,
                    learning_rate: 0.001,
                    beta1: 0.0,
                    beta2: 0.999,
                    on_change: move |opt| {
                        optimizers.write()[2] = opt;
                    }
                }
                OptimizerConfig {
                    optimizer_type: OptimizerType::Adam,
                    learning_rate: 0.001,
                    beta1: 0.9,
                    beta2: 0.999,
                    on_change: move |opt| {
                        optimizers.write()[3] = opt;
                    }
                }
            }

            // Train button
            div { class: "train-controls",
                button {
                    class: "train-button",
                    disabled: is_training(),
                    onclick: train_all,
                    if is_training() {
                        "Training... ({current_iteration()}/{max_iterations()})"
                    } else {
                        "Train All Optimizers"
                    }
                }
            }

            // Visualization
            LossLandscapeViz {
                loss_function: loss_function(),
                paths: optimizer_paths(),
                current_iteration: current_iteration(),
                is_animating: is_training()
            }

            // Metrics
            ConvergenceMetrics {
                paths: optimizer_paths(),
                optimizer_names: vec!["SGD", "Momentum", "RMSprop", "Adam"]
            }
        }
    }
}
```

### **Phase 3: Advanced Visualizations** (Week 3-4, 15-20 hours)

**Tasks:**
1. 2D Loss Landscape Heatmap
2. Optimizer Path Animation
3. Convergence Charts
4. Interactive Features

**Detailed Task Breakdown:**

#### Task 3.1: Loss Landscape Heatmap
```rust
#[component]
fn LossLandscapeViz(
    loss_function: LossFunction,
    paths: Vec<OptimizerPath>,
    current_iteration: usize,
    is_animating: bool,
) -> Element {
    // Compute heatmap grid
    let resolution = 50;
    let x_range = (-2.0, 2.0);
    let y_range = (-1.0, 3.0);

    let grid = use_memo(move || {
        let mut grid = vec![vec![0.0; resolution]; resolution];
        let mut max_loss = f64::NEG_INFINITY;

        for i in 0..resolution {
            for j in 0..resolution {
                let x = x_range.0 + (x_range.1 - x_range.0) * i as f64 / resolution as f64;
                let y = y_range.0 + (y_range.1 - y_range.0) * j as f64 / resolution as f64;

                let loss = loss_function.evaluate(x, y);
                grid[i][j] = loss;
                max_loss = max_loss.max(loss);
            }
        }

        (grid, max_loss)
    });

    rsx! {
        svg {
            class: "loss-landscape",
            view_box: "0 0 800 600",

            // Heatmap
            for (i, row) in grid().0.iter().enumerate() {
                for (j, &loss) in row.iter().enumerate() {
                    {
                        let intensity = (loss / grid().1 * 255.0).min(255.0) as u8;
                        let color = format!("rgb({}, {}, 255)", 255 - intensity, 255 - intensity);

                        rsx! {
                            rect {
                                x: "{i * 16}",
                                y: "{j * 12}",
                                width: "16",
                                height: "12",
                                fill: "{color}",
                                opacity: "0.8"
                            }
                        }
                    }
                }
            }

            // Optimizer paths
            {
                let colors = ["#ef4444", "#10b981", "#3b82f6", "#f59e0b"]; // red, green, blue, yellow
                let names = ["SGD", "Momentum", "RMSprop", "Adam"];

                paths.iter().enumerate().map(|(idx, path)| {
                    let color = colors[idx];
                    let name = names[idx];

                    rsx! {
                        g { class: "optimizer-path",
                            // Path line
                            polyline {
                                points: {
                                    path.points.iter()
                                        .map(|(x, y)| {
                                            let sx = ((x - x_range.0) / (x_range.1 - x_range.0) * 800.0) as i32;
                                            let sy = ((y - y_range.0) / (y_range.1 - y_range.0) * 600.0) as i32;
                                            format!("{},{}", sx, sy)
                                        })
                                        .collect::<Vec<_>>()
                                        .join(" ")
                                },
                                stroke: color,
                                stroke_width: "2",
                                fill: "none",
                                opacity: "0.8"
                            }

                            // Current position
                            if let Some(&(x, y)) = path.points.last() {
                                {
                                    let sx = (x - x_range.0) / (x_range.1 - x_range.0) * 800.0;
                                    let sy = (y - y_range.0) / (y_range.1 - y_range.0) * 600.0;

                                    rsx! {
                                        circle {
                                            cx: "{sx}",
                                            cy: "{sy}",
                                            r: "6",
                                            fill: color,
                                            stroke: "white",
                                            stroke_width: "2",
                                            class: if is_animating { "pulsing" } else { "" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }).collect::<Vec<_>>()
            }

            // Legend
            g { class: "legend", transform: "translate(650, 20)",
                {
                    let colors = ["#ef4444", "#10b981", "#3b82f6", "#f59e0b"];
                    let names = ["SGD", "Momentum", "RMSprop", "Adam"];

                    names.iter().enumerate().map(|(idx, &name)| {
                        let y = idx * 25;
                        rsx! {
                            g { transform: "translate(0, {y})",
                                circle { cx: "10", cy: "10", r: "5", fill: colors[idx] }
                                text { x: "20", y: "15", fill: "white", font_size: "12",
                                    "{name}"
                                }
                            }
                        }
                    }).collect::<Vec<_>>()
                }
            }
        }
    }
}
```

#### Task 3.2: Convergence Metrics
```rust
#[component]
fn ConvergenceMetrics(
    paths: Vec<OptimizerPath>,
    optimizer_names: Vec<&'static str>,
) -> Element {
    rsx! {
        div { class: "convergence-metrics",
            h3 { "Convergence Comparison" }

            table { class: "metrics-table",
                thead {
                    tr {
                        th { "Optimizer" }
                        th { "Iterations" }
                        th { "Final Loss" }
                        th { "Path Length" }
                        th { "Avg Gradient" }
                    }
                }
                tbody {
                    for (idx, (path, &name)) in paths.iter().zip(optimizer_names.iter()).enumerate() {
                        {
                            let iterations = path.points.len();
                            let final_loss = path.losses.last().copied().unwrap_or(f64::INFINITY);
                            let path_length: f64 = path.points.windows(2)
                                .map(|w| {
                                    let dx = w[1].0 - w[0].0;
                                    let dy = w[1].1 - w[0].1;
                                    (dx*dx + dy*dy).sqrt()
                                })
                                .sum();
                            let avg_gradient = path_length / iterations as f64;

                            // Highlight best performer
                            let is_best = paths.iter()
                                .filter_map(|p| p.losses.last())
                                .min_by(|a, b| a.partial_cmp(b).unwrap())
                                .map(|&min_loss| (final_loss - min_loss).abs() < 1e-6)
                                .unwrap_or(false);

                            rsx! {
                                tr { class: if is_best { "best-performer" } else { "" },
                                    td {
                                        span { class: "optimizer-badge", style: "background-color: {get_color(idx)}",
                                            "{name}"
                                        }
                                    }
                                    td { "{iterations}" }
                                    td { "{final_loss:.6}" }
                                    td { "{path_length:.2}" }
                                    td { "{avg_gradient:.4}" }
                                    if is_best {
                                        td { "⭐" }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Loss over iterations chart
            div { class: "loss-chart",
                h4 { "Loss Over Iterations" }
                svg {
                    view_box: "0 0 600 300",

                    // Axes
                    line { x1: "50", y1: "250", x2: "550", y2: "250", stroke: "#666", stroke_width: "2" }
                    line { x1: "50", y1: "50", x2: "50", y2: "250", stroke: "#666", stroke_width: "2" }

                    // Loss lines for each optimizer
                    {
                        let colors = ["#ef4444", "#10b981", "#3b82f6", "#f59e0b"];

                        paths.iter().enumerate().map(|(idx, path)| {
                            if path.losses.is_empty() {
                                return rsx! { };
                            }

                            let max_loss = paths.iter()
                                .flat_map(|p| p.losses.iter())
                                .cloned()
                                .fold(f64::NEG_INFINITY, f64::max);

                            let points: String = path.losses.iter().enumerate()
                                .map(|(i, &loss)| {
                                    let x = 50.0 + (i as f64 / path.losses.len() as f64) * 500.0;
                                    let y = 250.0 - (loss / max_loss) * 200.0;
                                    format!("{},{}", x, y)
                                })
                                .collect::<Vec<_>>()
                                .join(" ");

                            rsx! {
                                polyline {
                                    points: "{points}",
                                    stroke: colors[idx],
                                    stroke_width: "2",
                                    fill: "none"
                                }
                            }
                        }).collect::<Vec<_>>()
                    }
                }
            }
        }
    }
}
```

### **Phase 4: Educational Content & Polish** (Week 4, 10-15 hours)

**Tasks:**
1. Interactive tutorials
2. Educational annotations
3. Common failure modes demos
4. Documentation

**Detailed Task Breakdown:**

#### Task 4.1: Tutorial Component
```rust
#[component]
fn OptimizerTutorial() -> Element {
    let mut current_step = use_signal(|| 0);

    let steps = vec![
        ("Why Vanilla GD Fails", "Gradient descent with fixed learning rate struggles with ravines and saddle points."),
        ("Momentum to the Rescue", "Momentum accumulates velocity, smoothing out oscillations."),
        ("Adaptive Learning Rates", "RMSprop adapts per-parameter learning rates based on gradient history."),
        ("Adam: Best of Both", "Adam combines momentum with adaptive learning rates for robust performance."),
    ];

    rsx! {
        div { class: "tutorial",
            h2 { "Understanding Optimizers" }

            div { class: "tutorial-step",
                h3 { "{steps[current_step()].0}" }
                p { "{steps[current_step()].1}" }

                // Interactive demo for current step
                match current_step() {
                    0 => rsx! { VanillaGDFailureDemo {} },
                    1 => rsx! { MomentumDemo {} },
                    2 => rsx! { AdaptiveLRDemo {} },
                    3 => rsx! { AdamDemo {} },
                    _ => rsx! { }
                }
            }

            div { class: "tutorial-nav",
                button {
                    disabled: current_step() == 0,
                    onclick: move |_| current_step.set(current_step() - 1),
                    "← Previous"
                }
                span { "Step {current_step() + 1} of {steps.len()}" }
                button {
                    disabled: current_step() == steps.len() - 1,
                    onclick: move |_| current_step.set(current_step() + 1),
                    "Next →"
                }
            }
        }
    }
}
```

#### Task 4.2: Documentation
```rust
/// # Optimizer
///
/// Implements various gradient descent optimization algorithms.
///
/// ## Algorithms
///
/// ### SGD (Stochastic Gradient Descent)
/// ```text
/// θ = θ - α∇L(θ)
/// ```
/// Vanilla gradient descent. Simple but sensitive to learning rate.
///
/// ### Momentum
/// ```text
/// v = β₁·v + ∇L(θ)
/// θ = θ - α·v
/// ```
/// Accumulates velocity in consistent gradient directions. Helps escape ravines.
///
/// **Hyperparameters:**
/// - `β₁`: Momentum decay (typical: 0.9)
/// - Higher β₁ = more momentum = smoother path
///
/// ### RMSprop
/// ```text
/// s = β₂·s + (1-β₂)·(∇L(θ))²
/// θ = θ - α·∇L(θ)/√(s + ε)
/// ```
/// Adapts learning rate per parameter based on gradient magnitude.
///
/// **Hyperparameters:**
/// - `β₂`: RMS decay (typical: 0.999)
/// - `ε`: Numerical stability (typical: 1e-8)
///
/// ### Adam (Adaptive Moment Estimation)
/// ```text
/// m = β₁·m + (1-β₁)·∇L(θ)
/// v = β₂·v + (1-β₂)·(∇L(θ))²
/// m̂ = m / (1 - β₁ᵗ)
/// v̂ = v / (1 - β₂ᵗ)
/// θ = θ - α·m̂/√(v̂ + ε)
/// ```
/// Combines momentum with adaptive learning rates. Industry standard.
///
/// **Hyperparameters:**
/// - `β₁`: Momentum decay (typical: 0.9)
/// - `β₂`: RMS decay (typical: 0.999)
/// - `ε`: Numerical stability (typical: 1e-8)
///
/// ## Examples
///
/// ```rust
/// use neural_network::optimizer::{Optimizer, OptimizerType};
///
/// // Create Adam optimizer
/// let optimizer = Optimizer::adam(0.001, 0.9, 0.999, 1e-8);
///
/// // Or use builder pattern
/// let optimizer = Optimizer::new(OptimizerType::Adam, 0.001)
///     .with_beta1(0.9)
///     .with_beta2(0.999);
/// ```
///
/// ## When to Use Which Optimizer
///
/// | Optimizer | Best For | Avoid When |
/// |-----------|----------|------------|
/// | SGD | Simple problems, when you can tune LR | Ravines, saddle points |
/// | Momentum | Ravines, noisy gradients | Overshooting minima |
/// | RMSprop | RNNs, non-stationary objectives | Sparse gradients |
/// | Adam | General-purpose, default choice | When you need SGD's simplicity |
///
/// ## References
///
/// - Polyak (1964): "Some methods of speeding up the convergence"
/// - Hinton et al. (2012): RMSprop lecture
/// - Kingma & Ba (2014): "Adam: A Method for Stochastic Optimization"
```

## Acceptance Criteria

### Functional Requirements
- [ ] `Optimizer` struct implemented with 4 optimizer types (SGD, Momentum, RMSprop, Adam)
- [ ] State management for velocity and squared gradients
- [ ] Integration into `NeuralNetwork::backward()`
- [ ] Integration into `LinearRegressor::fit()`
- [ ] Web component with 4-panel optimizer comparison
- [ ] 2D loss landscape visualization with colored optimizer paths
- [ ] Hyperparameter controls (α, β₁, β₂) for each optimizer
- [ ] Real-time training with animated convergence
- [ ] Convergence metrics table

### Non-Functional Requirements
- [ ] Performance: 1000 iterations complete in <2 seconds (WASM)
- [ ] Responsiveness: UI updates every 10 iterations during training
- [ ] Accessibility: Keyboard navigation, screen reader support
- [ ] Browser compatibility: Chrome, Firefox, Safari (latest 2 versions)

### Quality Gates
- [ ] **Test Coverage**: >90% for `optimizer.rs`
  - Unit tests for each optimizer type
  - Convergence tests on standard benchmarks
  - Edge case handling (divergence, numerical stability)

- [ ] **Documentation**: Comprehensive rustdoc comments
  - API documentation for all public functions
  - Mathematical formulas in doc comments
  - Usage examples

- [ ] **Code Review**: Approved by maintainer
  - Clean separation of concerns
  - No performance regressions
  - Follows project conventions

## Algorithm Details

### Momentum (Polyak, 1964)
```
Initialize: v = 0
For each iteration:
    v = β₁·v + ∇L(θ)           # Exponential moving average of gradients
    θ = θ - α·v                 # Update parameters
```

**Key insight:** Accumulates velocity in consistent gradient directions, dampens oscillations

### RMSprop (Hinton et al., 2012)
```
Initialize: s = 0
For each iteration:
    s = β₂·s + (1-β₂)·(∇L(θ))²  # Exponential moving average of squared gradients
    θ = θ - α·∇L(θ)/√(s + ε)    # Adaptive per-parameter learning rates
```

**Key insight:** Large gradients get small learning rates, small gradients get large learning rates

### Adam (Kingma & Ba, 2014)
```
Initialize: m = 0, v = 0, t = 0
For each iteration:
    t = t + 1
    m = β₁·m + (1-β₁)·∇L(θ)            # First moment (mean)
    v = β₂·v + (1-β₂)·(∇L(θ))²         # Second moment (uncentered variance)
    m̂ = m / (1 - β₁ᵗ)                   # Bias correction
    v̂ = v / (1 - β₂ᵗ)                   # Bias correction
    θ = θ - α·m̂/√(v̂ + ε)               # Update
```

**Key insight:** Combines momentum + adaptive learning rates + bias correction for warm start

## Risk Analysis & Mitigation

### Technical Risks

**Risk 1: WASM Performance**
- **Probability**: Medium
- **Impact**: High (sluggish UI defeats educational purpose)
- **Mitigation**:
  - Profile with `wasm-pack` profiler
  - Use `f32` instead of `f64` where precision allows
  - Implement progressive rendering (update every 10 iterations)
  - Add "fast mode" option (skip visualization during training)

**Risk 2: Numerical Instability**
- **Probability**: Medium
- **Impact**: Medium (divergence, NaN values)
- **Mitigation**:
  - Gradient clipping (max norm = 1.0)
  - Epsilon term in denominators (1e-8)
  - Input validation on hyperparameters
  - Comprehensive unit tests with edge cases

**Risk 3: UI Complexity**
- **Probability**: Low
- **Impact**: Medium (overwhelming for beginners)
- **Mitigation**:
  - Progressive disclosure (advanced settings collapsed)
  - Sensible defaults (β₁=0.9, β₂=0.999, α=0.001)
  - Preset configurations ("Fast Convergence", "Stable Training")
  - Tooltips explaining every parameter

### Educational Risks

**Risk 4: Too Abstract**
- **Probability**: Low
- **Impact**: High (users don't understand the value)
- **Mitigation**:
  - Start with concrete problem ("This is why your NN won't train")
  - Show failure mode first (vanilla GD oscillating)
  - Then show solution (Adam converging smoothly)
  - Real-world use case examples

## References & Research

### Internal References
- **Existing gradient descent**: `/Users/brunodossantos/Code/brunoml/cargo_workspace/linear_regression/src/lib.rs:33-77` (fit method)
- **Neural network training**: `/Users/brunodossantos/Code/brunoml/cargo_workspace/neural_network/src/lib.rs:278-309` (fit method)
- **Web visualization patterns**: `/Users/brunodossantos/Code/brunoml/cargo_workspace/web/src/components/showcase.rs:486-967` (GradientDescentDemo)

### External References

**Academic Papers:**
- Polyak, B. T. (1964). "Some methods of speeding up the convergence of iteration methods"
- Hinton, G. et al. (2012). "RMSprop: Divide the gradient by a running average of its recent magnitude"
- Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization" [[arXiv:1412.6980](https://arxiv.org/abs/1412.6980)]

**Industry Best Practices:**
- Sebastian Ruder. (2016). "An overview of gradient descent optimization algorithms" [[Blog](https://ruder.io/optimizing-gradient-descent/)]
- TensorFlow Optimizers Guide [[Docs](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)]
- PyTorch Optimizers [[Docs](https://pytorch.org/docs/stable/optim.html)]

**Visualization Inspiration:**
- Alec Radford's visualization [[Blog](https://imgur.com/a/Hqolp)]
- Distill.pub: "Why Momentum Really Works" [[Article](https://distill.pub/2017/momentum/)]

### Related Work
- **TensorFlow Playground**: Excellent but limited to pre-defined scenarios
- **CNN Explainer**: Beautiful visualization but for pre-trained models only
- **This Project's Advantage**: Interactive training on custom data with full transparency

## Future Considerations

### Phase 2 Features (Post-Launch)
- **Learning Rate Schedules**: Step decay, exponential decay, cosine annealing
- **Adaptive Optimizers**: AdaGrad, AdaDelta, Nadam
- **Second-Order Methods**: L-BFGS, conjugate gradient
- **3D Loss Landscapes**: Interactive WebGL visualization

### Extensibility
- **Export Functionality**: Save optimizer paths as JSON/CSV
- **Share Configurations**: URL parameters for hyperparameters
- **Custom Loss Functions**: User-defined mathematical expressions
- **Benchmark Suite**: Compare optimizers across multiple standard functions

### Integration Opportunities
- **Multi-Class Classification**: Use optimizers for Softmax+CrossEntropy
- **Regularization**: L1/L2 with different optimizers
- **CNNs**: Adam for convolutional layer training (future)
