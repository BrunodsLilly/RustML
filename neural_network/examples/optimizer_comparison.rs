//! Optimizer Comparison on the Rosenbrock Function
//!
//! This example demonstrates the behavioral differences between four gradient descent optimizers:
//! - SGD (Stochastic Gradient Descent)
//! - Momentum
//! - RMSprop
//! - Adam
//!
//! We use the famous Rosenbrock function as our test case:
//!   f(x,y) = (1-x)² + 100(y-x²)²
//!
//! This function is notorious for its narrow, curved valley that makes optimization challenging.
//! The global minimum is at (1, 1) where f(1,1) = 0.
//!
//! Run this example with:
//!   cargo run --example optimizer_comparison -p neural_network
//!
//! This is pedagogical code in the spirit of Andrej Karpathy - we'll explain what's happening
//! and why each optimizer behaves the way it does.

use linear_algebra::matrix::Matrix;
use neural_network::optimizer::Optimizer;

// ============================================================================
// The Rosenbrock Function
// ============================================================================

/// The Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
///
/// This is one of the most famous test functions in optimization. It has:
/// - A global minimum at (1, 1) with f(1,1) = 0
/// - A narrow, curved valley that's easy to find but hard to optimize along
/// - Steep walls on either side of the valley
///
/// These characteristics make it a great benchmark for optimizer performance.
fn rosenbrock(x: f64, y: f64) -> f64 {
    (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2)
}

/// Gradient of the Rosenbrock function
///
/// Computed via calculus:
///   ∂f/∂x = -2(1-x) - 400x(y-x²)
///   ∂f/∂y = 200(y-x²)
fn rosenbrock_gradient(x: f64, y: f64) -> (f64, f64) {
    let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x.powi(2));
    let dy = 200.0 * (y - x.powi(2));
    (dx, dy)
}

// ============================================================================
// Visualization Helpers
// ============================================================================

/// Print a simple ASCII art visualization of the optimizer's path
fn print_path_ascii(name: &str, path: &[(f64, f64)]) {
    println!("\n{} Path Visualization:", name);
    println!("(Arrows show direction of movement)\n");

    // Sample key points along the path
    let step_size = (path.len() / 10).max(1);
    for i in (0..path.len()).step_by(step_size) {
        let (x, y) = path[i];
        let loss = rosenbrock(x, y);

        // Show direction if not the last point
        let direction = if i + step_size < path.len() {
            let (next_x, next_y) = path[i + step_size];
            let dx = next_x - x;
            let dy = next_y - y;

            if dx.abs() > dy.abs() {
                if dx > 0.0 {
                    "→"
                } else {
                    "←"
                }
            } else {
                if dy > 0.0 {
                    "↑"
                } else {
                    "↓"
                }
            }
        } else {
            "★" // Star for the end point
        };

        println!(
            "  Iter {:4}: ({:7.4}, {:7.4})  loss={:10.6}  {}",
            i, x, y, loss, direction
        );
    }
}

// ============================================================================
// Main Comparison
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║      Optimizer Comparison: Rosenbrock Function Benchmark      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("Problem: Minimize f(x,y) = (1-x)² + 100(y-x²)²");
    println!("Starting point: (-1.0, 1.0)");
    println!("Target: (1.0, 1.0) where f = 0");
    println!("Challenge: Narrow curved valley requires smart navigation\n");

    // Configuration for each optimizer
    // Note: Learning rates are different because optimizers have different sensitivities
    let optimizers = vec![
        ("SGD", Optimizer::sgd(0.0001), "Vanilla gradient descent"),
        (
            "Momentum",
            Optimizer::momentum(0.0001, 0.9),
            "Accumulates velocity (β₁=0.9)",
        ),
        (
            "RMSprop",
            Optimizer::rmsprop(0.001, 0.999, 1e-8),
            "Adaptive per-parameter rates",
        ),
        (
            "Adam",
            Optimizer::adam(0.001, 0.9, 0.999, 1e-8),
            "Momentum + Adaptive rates",
        ),
    ];

    let starting_point = (-1.0, 1.0);
    let target_point = (1.0, 1.0);
    let max_iterations = 2000;

    println!("═══════════════════════════════════════════════════════════════\n");

    for (name, mut opt, description) in optimizers {
        println!("─────────────────────────────────────────────────────────────");
        println!("Optimizer: {} - {}", name, description);
        println!("─────────────────────────────────────────────────────────────\n");

        // Initialize at starting point
        let mut weights = Matrix::from_vec(vec![starting_point.0, starting_point.1], 1, 2)
            .expect("Failed to create matrix");

        let layer_shapes = vec![(1, 2)];

        // Track the path for visualization
        let mut path = vec![starting_point];
        let initial_loss = rosenbrock(starting_point.0, starting_point.1);

        // Run optimization
        for iteration in 0..max_iterations {
            let x = weights[(0, 0)];
            let y = weights[(0, 1)];

            // Compute gradient
            let (dx, dy) = rosenbrock_gradient(x, y);
            let gradient =
                Matrix::from_vec(vec![dx, dy], 1, 2).expect("Failed to create gradient matrix");

            // Update weights
            opt.update_weights(0, &gradient, &mut weights, &layer_shapes);

            // Record path (sample every 10 iterations to save memory)
            if iteration % 10 == 0 {
                path.push((weights[(0, 0)], weights[(0, 1)]));
            }
        }

        // Final results
        let final_x = weights[(0, 0)];
        let final_y = weights[(0, 1)];
        let final_loss = rosenbrock(final_x, final_y);
        let distance_to_target =
            ((final_x - target_point.0).powi(2) + (final_y - target_point.1).powi(2)).sqrt();

        // Calculate path length (total distance traveled)
        let path_length: f64 = path
            .windows(2)
            .map(|w| {
                let dx = w[1].0 - w[0].0;
                let dy = w[1].1 - w[0].1;
                (dx * dx + dy * dy).sqrt()
            })
            .sum();

        // Print results
        println!("Results after {} iterations:", max_iterations);
        println!(
            "  Initial:  ({:7.4}, {:7.4})  loss = {:.6}",
            starting_point.0, starting_point.1, initial_loss
        );
        println!(
            "  Final:    ({:7.4}, {:7.4})  loss = {:.6}",
            final_x, final_y, final_loss
        );
        println!(
            "  Target:   ({:7.4}, {:7.4})  loss = {:.6}",
            target_point.0, target_point.1, 0.0
        );
        println!();
        println!(
            "  Loss reduction: {:.2}x  ({:.6} → {:.6})",
            initial_loss / final_loss,
            initial_loss,
            final_loss
        );
        println!("  Distance to target: {:.6}", distance_to_target);
        println!("  Path length: {:.4}", path_length);

        // Determine convergence quality
        let quality = if distance_to_target < 0.01 {
            "Excellent ✓"
        } else if distance_to_target < 0.1 {
            "Good"
        } else if distance_to_target < 0.5 {
            "Moderate"
        } else {
            "Poor"
        };
        println!("  Convergence quality: {}", quality);

        // Show path visualization
        print_path_ascii(name, &path);

        println!();
    }

    println!("═══════════════════════════════════════════════════════════════\n");
    println!("Key Insights:");
    println!();
    println!("1. SGD: Slow but steady. Large learning rates cause oscillations,");
    println!("   small rates converge slowly. Very sensitive to learning rate.");
    println!();
    println!("2. Momentum: Accelerates in consistent directions, dampens");
    println!("   oscillations. Good for ravines but can overshoot.");
    println!();
    println!("3. RMSprop: Adapts learning rate per parameter. Handles different");
    println!("   gradient scales well. Good for non-stationary problems.");
    println!();
    println!("4. Adam: Combines momentum + adaptive rates + bias correction.");
    println!("   Generally fastest and most robust. Industry standard.");
    println!();
    println!("Try adjusting learning rates to see how each optimizer responds!");
    println!("═══════════════════════════════════════════════════════════════\n");
}
