use linear_algebra::{matrix::Matrix, vectors::Vector};
use linear_regression::LinearRegressor;

/// Demonstrates linear regression with gradient descent on simple 1D data
/// Goal: Learn the function y = 2x + 1
fn main() {
    println!("=== Linear Regression with Gradient Descent ===\n");

    // Create training data: y = 2x + 1
    println!("Generating training data: y = 2x + 1");
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y_data: Vec<f64> = x_data.iter().map(|x| 2.0 * x + 1.0).collect();

    println!("Training samples: {}", x_data.len());
    for i in 0..3 {
        println!("  x={}, y={}", x_data[i], y_data[i]);
    }
    println!("  ...\n");

    // Create matrix and vector
    let X = Matrix::from_vec(x_data, 10, 1).unwrap();
    let y = Vector { data: y_data };

    // Create and train model
    println!("Creating model with learning rate = 0.01");
    let mut model = LinearRegressor::new(0.01);

    println!("\nTraining for 1000 iterations...\n");
    model.fit(&X, &y, 1000);

    // Display results
    println!("\n=== Training Results ===");
    println!("Learned weight: {:.6}", model.weights.data[0]);
    println!("Learned bias: {:.6}", model.bias);
    println!("True weight: 2.000000");
    println!("True bias: 1.000000\n");

    // Test predictions
    println!("=== Testing Predictions ===");
    let test_x = vec![11.0, 12.0, 15.0];
    for &x_val in &test_x {
        let test_vec = Vector { data: vec![x_val] };
        let pred = model.predict_single(&test_vec);
        let true_val = 2.0 * x_val + 1.0;
        let error = (pred - true_val).abs();
        println!(
            "x={:5.1} | Predicted: {:7.3} | True: {:7.3} | Error: {:.6}",
            x_val, pred, true_val, error
        );
    }

    // Show cost reduction
    println!("\n=== Cost History (every 100 iterations) ===");
    for (i, cost) in model
        .training_history
        .iter()
        .enumerate()
        .step_by(100)
        .take(10)
    {
        println!("Iteration {:4}: Cost = {:.6}", i, cost);
    }

    println!("\n✓ Training complete! The model learned the linear relationship.");
}
