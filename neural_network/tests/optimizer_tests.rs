//! Comprehensive tests for gradient descent optimizers
//!
//! Following the Karpathy philosophy: clear, pedagogical tests that teach us
//! how these optimizers actually behave, not just that they compile.

use linear_algebra::{matrix::Matrix, vectors::Vector};
use neural_network::optimizer::{Optimizer, OptimizerType};

// ===================================================================
// Helper Functions
// ===================================================================

/// Create a matrix - helper to avoid unwrapping Results everywhere
fn mat(data: Vec<f64>, rows: usize, cols: usize) -> Matrix<f64> {
    Matrix::from_vec(data, rows, cols).expect("Invalid matrix dimensions")
}

/// Create a vector - helper for cleaner tests
fn vec_f64(data: Vec<f64>) -> Vector<f64> {
    Vector { data }
}

// ===================================================================
// Test 1: Creation and Basic Properties
// ===================================================================

#[test]
fn test_optimizer_creation() {
    let sgd = Optimizer::sgd(0.01);
    assert_eq!(sgd.optimizer_type(), OptimizerType::SGD);
    assert_eq!(sgd.learning_rate(), 0.01);

    let momentum = Optimizer::momentum(0.01, 0.9);
    assert_eq!(momentum.optimizer_type(), OptimizerType::Momentum);
    assert_eq!(momentum.beta1(), 0.9);

    let rmsprop = Optimizer::rmsprop(0.001, 0.999, 1e-8);
    assert_eq!(rmsprop.optimizer_type(), OptimizerType::RMSprop);
    assert_eq!(rmsprop.beta2(), 0.999);

    let adam = Optimizer::adam(0.001, 0.9, 0.999, 1e-8);
    assert_eq!(adam.optimizer_type(), OptimizerType::Adam);
    assert_eq!(adam.beta1(), 0.9);
    assert_eq!(adam.beta2(), 0.999);
}

// ===================================================================
// Test 2: SGD - The Baseline
// ===================================================================

#[test]
fn test_sgd_simple_update() {
    // Test the most basic case: single weight, single gradient step
    // f(x) = x^2, gradient = 2x
    // At x=1, gradient=2, so new x = 1 - 0.1*2 = 0.8

    let mut opt = Optimizer::sgd(0.1);
    let mut weights = mat(vec![1.0], 1, 1);
    let gradient = mat(vec![2.0], 1, 1);
    let layer_shapes = vec![(1, 1)];

    opt.update_weights(0, &gradient, &mut weights, &layer_shapes);

    assert!(
        (weights[(0, 0)] - 0.8).abs() < 1e-10,
        "SGD: x_new = x_old - lr*grad = 1.0 - 0.1*2.0 = 0.8"
    );
}

#[test]
fn test_sgd_converges() {
    // Minimize f(x) = x^2 starting from x=2
    let mut opt = Optimizer::sgd(0.1);
    let mut weights = mat(vec![2.0], 1, 1);
    let layer_shapes = vec![(1, 1)];

    for _ in 0..50 {
        let x = weights[(0, 0)];
        let gradient = mat(vec![2.0 * x], 1, 1); // df/dx = 2x
        opt.update_weights(0, &gradient, &mut weights, &layer_shapes);
    }

    assert!(
        weights[(0, 0)].abs() < 0.01,
        "After 50 steps, SGD should get close to x=0"
    );
}

#[test]
fn test_sgd_bias_update() {
    let mut opt = Optimizer::sgd(0.1);
    let mut bias = vec_f64(vec![1.0, 2.0, 3.0]);
    let gradient = vec_f64(vec![0.5, 1.0, 1.5]);
    let layer_shapes = vec![(3, 2)];

    opt.update_bias(0, &gradient, &mut bias, &layer_shapes);

    // b_new = b_old - lr * grad
    assert!((bias.data[0] - 0.95).abs() < 1e-10); // 1.0 - 0.1*0.5
    assert!((bias.data[1] - 1.90).abs() < 1e-10); // 2.0 - 0.1*1.0
    assert!((bias.data[2] - 2.85).abs() < 1e-10); // 3.0 - 0.1*1.5
}

// ===================================================================
// Test 3: Momentum - Builds Velocity
// ===================================================================

#[test]
fn test_momentum_accelerates() {
    // Key insight: With consistent gradients, momentum should take bigger steps over time
    let mut opt = Optimizer::momentum(0.1, 0.9);
    let mut weights = mat(vec![1.0], 1, 1);
    let gradient = mat(vec![1.0], 1, 1); // Constant gradient
    let layer_shapes = vec![(1, 1)];

    let x0 = weights[(0, 0)];
    opt.update_weights(0, &gradient, &mut weights, &layer_shapes);
    let step1 = (x0 - weights[(0, 0)]).abs();

    let x1 = weights[(0, 0)];
    opt.update_weights(0, &gradient, &mut weights, &layer_shapes);
    let step2 = (x1 - weights[(0, 0)]).abs();

    let x2 = weights[(0, 0)];
    opt.update_weights(0, &gradient, &mut weights, &layer_shapes);
    let step3 = (x2 - weights[(0, 0)]).abs();

    println!("Momentum steps: {:.6}, {:.6}, {:.6}", step1, step2, step3);

    // Each step should be larger (momentum building up)
    assert!(step2 > step1, "Step 2 > step 1 (momentum accelerating)");
    assert!(step3 > step2, "Step 3 > step 2 (momentum accelerating)");
}

// ===================================================================
// Test 4: RMSprop - Adaptive Learning Rates
// ===================================================================

#[test]
fn test_rmsprop_adapts() {
    // RMSprop should give different effective learning rates to different dimensions
    let mut opt = Optimizer::rmsprop(0.1, 0.9, 1e-8);
    let mut weights = mat(vec![1.0, 1.0], 1, 2); // Two parameters
    let layer_shapes = vec![(1, 2)];

    // Run 10 steps with very different gradient magnitudes
    for _ in 0..10 {
        let gradient = mat(vec![10.0, 0.1], 1, 2); // 100x difference
        opt.update_weights(0, &gradient, &mut weights, &layer_shapes);
    }

    let change0 = (1.0 - weights[(0, 0)]).abs();
    let change1 = (1.0 - weights[(0, 1)]).abs();

    println!(
        "RMSprop changes: dim0={:.6}, dim1={:.6}, ratio={:.2}",
        change0,
        change1,
        change0 / change1
    );

    // Despite 100x gradient difference, changes shouldn't differ by 100x
    assert!(
        change0 / change1 < 50.0,
        "RMSprop should adapt learning rates to prevent huge imbalance"
    );
}

// ===================================================================
// Test 5: Adam - Best of Both Worlds
// ===================================================================

#[test]
fn test_adam_bias_correction() {
    // Adam's bias correction prevents tiny first steps
    let mut opt = Optimizer::adam(0.001, 0.9, 0.999, 1e-8);
    let mut weights = mat(vec![1.0], 1, 1);
    let gradient = mat(vec![1.0], 1, 1);
    let layer_shapes = vec![(1, 1)];

    let x0 = weights[(0, 0)];
    opt.update_weights(0, &gradient, &mut weights, &layer_shapes);
    let step1 = (x0 - weights[(0, 0)]).abs();

    println!("Adam first step: {:.8}", step1);

    // Without bias correction, this would be ~0.0001
    // With bias correction, it should be reasonable
    assert!(
        step1 > 0.0001,
        "Bias correction should prevent tiny first step"
    );
}

// ===================================================================
// Test 6: Rosenbrock Function - Classic Benchmark
// ===================================================================

/// Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2
/// Minimum at (1,1), famous for its narrow curved valley
fn rosenbrock(x: f64, y: f64) -> f64 {
    (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2)
}

fn rosenbrock_grad(x: f64, y: f64) -> (f64, f64) {
    let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x.powi(2));
    let dy = 200.0 * (y - x.powi(2));
    (dx, dy)
}

#[test]
fn test_all_optimizers_on_rosenbrock() {
    println!("\n=== Rosenbrock Benchmark ===");
    println!("Start: (-1, 1), Target: (1, 1)\n");

    let tests = vec![
        ("SGD", Optimizer::sgd(0.0001)),
        ("Momentum", Optimizer::momentum(0.0001, 0.9)),
        ("RMSprop", Optimizer::rmsprop(0.001, 0.999, 1e-8)),
        ("Adam", Optimizer::adam(0.001, 0.9, 0.999, 1e-8)),
    ];

    for (name, mut opt) in tests {
        let mut weights = mat(vec![-1.0, 1.0], 1, 2);
        let layer_shapes = vec![(1, 2)];

        let loss0 = rosenbrock(weights[(0, 0)], weights[(0, 1)]);

        // 1000 iterations
        for _ in 0..1000 {
            let x = weights[(0, 0)];
            let y = weights[(0, 1)];
            let (dx, dy) = rosenbrock_grad(x, y);

            let grad = mat(vec![dx, dy], 1, 2);
            opt.update_weights(0, &grad, &mut weights, &layer_shapes);
        }

        let x_final = weights[(0, 0)];
        let y_final = weights[(0, 1)];
        let loss_final = rosenbrock(x_final, y_final);
        let dist = ((x_final - 1.0).powi(2) + (y_final - 1.0).powi(2)).sqrt();

        println!(
            "{:8} → ({:.4}, {:.4}), loss={:.4}, dist={:.4}",
            name, x_final, y_final, loss_final, dist
        );

        // All should make progress
        assert!(loss_final < loss0, "{} should reduce loss", name);
    }
}

// ===================================================================
// Test 7: Edge Cases
// ===================================================================

#[test]
fn test_zero_gradient() {
    let mut opt = Optimizer::adam(0.1, 0.9, 0.999, 1e-8);
    let mut weights = mat(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let original = weights.clone();
    let zero_grad = mat(vec![0.0, 0.0, 0.0, 0.0], 2, 2);
    let layer_shapes = vec![(2, 2)];

    opt.update_weights(0, &zero_grad, &mut weights, &layer_shapes);

    // Weights shouldn't change with zero gradient
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(weights[(i, j)], original[(i, j)]);
        }
    }
}

#[test]
fn test_optimizer_reset() {
    let mut opt = Optimizer::adam(0.001, 0.9, 0.999, 1e-8);
    let mut weights = mat(vec![1.0], 1, 1);
    let grad = mat(vec![1.0], 1, 1);
    let layer_shapes = vec![(1, 1)];

    // Build up state
    for _ in 0..5 {
        opt.update_weights(0, &grad, &mut weights, &layer_shapes);
    }

    let final_weight_before_reset = weights[(0, 0)];

    // Reset
    opt.reset();

    // After reset, start fresh from same initial position
    weights = mat(vec![1.0], 1, 1);
    let mut weights2 = mat(vec![1.0], 1, 1);
    let mut opt_fresh = Optimizer::adam(0.001, 0.9, 0.999, 1e-8);

    opt.update_weights(0, &grad, &mut weights, &layer_shapes);
    opt_fresh.update_weights(0, &grad, &mut weights2, &layer_shapes);

    println!("Reset optimizer weight: {:.10}", weights[(0, 0)]);
    println!("Fresh optimizer weight: {:.10}", weights2[(0, 0)]);

    assert!(
        (weights[(0, 0)] - weights2[(0, 0)]).abs() < 1e-10,
        "Reset optimizer should match fresh optimizer"
    );

    // Also verify weights actually changed from the pre-reset value
    assert!(
        (weights[(0, 0)] - final_weight_before_reset).abs() > 0.001,
        "After reset, optimizer should start fresh, not continue from old state"
    );
}

#[test]
fn test_learning_rate_change() {
    let mut opt = Optimizer::sgd(0.1);
    opt.set_learning_rate(0.01);
    assert_eq!(opt.learning_rate(), 0.01);

    let mut weights = mat(vec![1.0], 1, 1);
    let grad = mat(vec![1.0], 1, 1);
    let layer_shapes = vec![(1, 1)];

    opt.update_weights(0, &grad, &mut weights, &layer_shapes);

    // Should use new LR: 1.0 - 0.01*1.0 = 0.99
    assert!((weights[(0, 0)] - 0.99).abs() < 1e-10);
}

// ===================================================================
// Test 8: Multi-Layer Networks
// ===================================================================

#[test]
fn test_multi_layer_updates() {
    let mut opt = Optimizer::adam(0.001, 0.9, 0.999, 1e-8);

    let layer_shapes = vec![(4, 3), (2, 4), (1, 2)];

    let mut weights = vec![
        mat(vec![1.0; 12], 4, 3),
        mat(vec![2.0; 8], 2, 4),
        mat(vec![3.0; 2], 1, 2),
    ];

    let grads = vec![
        mat(vec![0.1; 12], 4, 3),
        mat(vec![0.1; 8], 2, 4),
        mat(vec![0.1; 2], 1, 2),
    ];

    let originals: Vec<_> = weights.iter().map(|w| w.clone()).collect();

    // Update all layers
    for i in 0..3 {
        opt.update_weights(i, &grads[i], &mut weights[i], &layer_shapes);
    }

    // All weights should have changed
    for layer in 0..3 {
        let mut changed = false;
        for i in 0..weights[layer].rows {
            for j in 0..weights[layer].cols {
                if (weights[layer][(i, j)] - originals[layer][(i, j)]).abs() > 1e-10 {
                    changed = true;
                }
            }
        }
        assert!(changed, "Layer {} weights should have updated", layer);
    }
}
