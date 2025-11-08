use linear_algebra::{matrix::Matrix, vectors::Vector};
use neural_network::{NeuralNetwork, activation::ActivationType};

fn main() {
    println!("=== Neural Network XOR Problem Demo ===\n");

    // The XOR problem: Classic non-linearly separable problem
    // Input  | Output
    // -------|-------
    // 0 0    | 0
    // 0 1    | 1
    // 1 0    | 1
    // 1 1    | 0

    println!("Problem: Learn the XOR function");
    println!("XOR is NOT linearly separable, requiring hidden layers\n");

    // Training data
    let X = Matrix::from_vec(
        vec![
            0.0, 0.0,  // Input: [0, 0]
            0.0, 1.0,  // Input: [0, 1]
            1.0, 0.0,  // Input: [1, 0]
            1.0, 1.0,  // Input: [1, 1]
        ],
        4,  // 4 samples
        2,  // 2 features
    ).unwrap();

    let y = Matrix::from_vec(
        vec![
            0.0,  // 0 XOR 0 = 0
            1.0,  // 0 XOR 1 = 1
            1.0,  // 1 XOR 0 = 1
            0.0,  // 1 XOR 1 = 0
        ],
        4,  // 4 samples
        1,  // 1 output
    ).unwrap();

    println!("Training Data:");
    for i in 0..4 {
        println!("  Input: [{:.0}, {:.0}] -> Target: {:.0}",
                 X[(i, 0)], X[(i, 1)], y[(i, 0)]);
    }
    println!();

    // Create neural network
    // Architecture: 2 inputs -> 4 hidden neurons (Tanh) -> 1 output (Sigmoid)
    println!("Network Architecture:");
    println!("  Input Layer:  2 neurons");
    println!("  Hidden Layer: 4 neurons (Tanh activation)");
    println!("  Output Layer: 1 neuron  (Sigmoid activation)");
    println!("  Learning Rate: 0.5");
    println!();

    let mut nn = NeuralNetwork::new(
        &[2, 4, 1],  // Layer sizes
        &[ActivationType::Tanh, ActivationType::Sigmoid],  // Activations
        0.5,  // Learning rate
    );

    // Train the network
    println!("Training for 1000 epochs...");
    let epochs = 1000;
    let snapshot_interval = 200;  // Save snapshots every 200 epochs

    nn.fit(&X, &y, epochs, snapshot_interval);

    // Show training progress
    println!("\nTraining Progress:");
    let history_len = nn.history.losses.len();
    let checkpoints = vec![0, history_len / 4, history_len / 2, 3 * history_len / 4, history_len - 1];

    for &idx in &checkpoints {
        if idx < history_len {
            println!("  Epoch {:4}: Loss = {:.6}, Accuracy = {:.2}%",
                     idx + 1,
                     nn.history.losses[idx],
                     nn.history.accuracies[idx] * 100.0);
        }
    }
    println!();

    // Test predictions
    println!("Final Predictions:");
    let predictions = nn.predict(&X);

    println!("┌───────────┬────────┬────────────┬─────────┐");
    println!("│   Input   │ Target │ Prediction │ Correct │");
    println!("├───────────┼────────┼────────────┼─────────┤");

    for i in 0..4 {
        let input_0 = X[(i, 0)] as i32;
        let input_1 = X[(i, 1)] as i32;
        let target = y[(i, 0)];
        let prediction = predictions[(i, 0)];
        let predicted_class = if prediction > 0.5 { 1 } else { 0 };
        let correct = if (predicted_class as f64 - target).abs() < 0.1 { "✓" } else { "✗" };

        println!("│ [{}, {}]      │   {:.0}    │   {:.4}    │    {}    │",
                 input_0, input_1, target, prediction, correct);
    }

    println!("└───────────┴────────┴────────────┴─────────┘");
    println!();

    // Analysis
    let final_loss = nn.history.losses.last().unwrap();
    let final_accuracy = nn.history.accuracies.last().unwrap();

    println!("Performance Summary:");
    println!("  Final Loss:     {:.6}", final_loss);
    println!("  Final Accuracy: {:.2}%", final_accuracy * 100.0);

    if *final_accuracy >= 0.95 {
        println!("  Status: ✓ Successfully learned XOR function!");
    } else if *final_accuracy >= 0.75 {
        println!("  Status: ⚠ Partially learned (may need more training)");
    } else {
        println!("  Status: ✗ Failed to learn (try different hyperparameters)");
    }

    println!();
    println!("=== Demo Complete ===");
}
