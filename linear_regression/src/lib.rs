use std::path::Path;

use linear_algebra::{matrix::Matrix, vectors::Vector};
use loader::read;

/// Linear Regression model using gradient descent
/// Implements y = Xw + b where X is input matrix, w is weights, b is bias
#[derive(Debug, Clone)]
pub struct LinearRegressor {
    pub weights: Vector<f64>,
    pub bias: f64,
    pub learning_rate: f64,
    pub training_history: Vec<f64>, // Track cost over iterations
}

impl Default for LinearRegressor {
    fn default() -> Self {
        LinearRegressor {
            weights: Vector { data: vec![] },
            bias: 0.0,
            learning_rate: 0.01,
            training_history: vec![],
        }
    }
}

impl LinearRegressor {
    /// Create a new LinearRegressor with specified learning rate
    pub fn new(learning_rate: f64) -> Self {
        LinearRegressor {
            learning_rate,
            ..Default::default()
        }
    }

    /// Initialize weights with zeros for given number of features
    pub fn initialize(&mut self, n_features: usize) {
        self.weights = Vector {
            data: vec![0.0; n_features],
        };
        self.bias = 0.0;
        self.training_history.clear();
    }

    /// Initialize weights with small random values (better for convergence)
    pub fn initialize_random(&mut self, n_features: usize, scale: f64) {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hash, Hasher};

        // Simple pseudo-random initialization
        let s = RandomState::new();
        let mut weights = Vec::with_capacity(n_features);

        for i in 0..n_features {
            let mut hasher = s.build_hasher();
            i.hash(&mut hasher);
            let hash_val = hasher.finish();
            let val = ((hash_val % 10000) as f64 / 10000.0 - 0.5) * scale;
            weights.push(val);
        }

        self.weights = Vector { data: weights };
        self.bias = 0.0;
        self.training_history.clear();
    }

    /// Predict using a single feature vector
    /// Returns: dot(x, weights) + bias
    pub fn predict_single(&self, x: &Vector<f64>) -> f64 {
        assert_eq!(
            x.data.len(),
            self.weights.data.len(),
            "Input features must match weight dimensions"
        );
        x.dot(&self.weights) + self.bias
    }

    /// Predict for a batch of samples using Matrix
    /// X: Matrix of shape (n_samples, n_features)
    /// Returns: Vector of predictions of length n_samples
    pub fn predict(&self, X: &Matrix<f64>) -> Vector<f64> {
        assert_eq!(
            X.cols,
            self.weights.data.len(),
            "Input features must match weight dimensions"
        );

        // Compute Xw
        let mut predictions = vec![0.0; X.rows];
        for i in 0..X.rows {
            let mut sum = 0.0;
            for j in 0..X.cols {
                sum += X[(i, j)] * self.weights.data[j];
            }
            predictions[i] = sum + self.bias;
        }

        Vector { data: predictions }
    }

    /// Calculate Mean Squared Error cost
    /// predictions: predicted values
    /// targets: actual target values
    pub fn cost(predictions: &Vector<f64>, targets: &Vector<f64>) -> f64 {
        assert_eq!(
            predictions.data.len(),
            targets.data.len(),
            "Predictions and targets must have the same length"
        );

        let n = predictions.data.len() as f64;
        let mut sum_squared_error = 0.0;

        for (pred, target) in predictions.data.iter().zip(targets.data.iter()) {
            let error = pred - target;
            sum_squared_error += error * error;
        }

        sum_squared_error / (2.0 * n)
    }

    /// Train the model using batch gradient descent
    /// X: Training data matrix (n_samples, n_features)
    /// y: Target vector (n_samples,)
    /// iterations: Number of gradient descent iterations
    pub fn fit(&mut self, X: &Matrix<f64>, y: &Vector<f64>, iterations: usize) {
        assert_eq!(X.rows, y.data.len(), "X rows must match y length");

        // Initialize weights if not already initialized
        if self.weights.data.is_empty() {
            self.initialize(X.cols);
        }

        let m = X.rows as f64; // number of samples

        // Gradient descent
        for iter in 0..iterations {
            // Forward pass: compute predictions
            let predictions = self.predict(X);

            // Compute cost and store in history
            let current_cost = Self::cost(&predictions, y);
            self.training_history.push(current_cost);

            // Print progress every 100 iterations
            if iter % 100 == 0 {
                println!("Iteration {}: Cost = {:.6}", iter, current_cost);
            }

            // Compute errors
            let errors = predictions - y.clone();

            // Compute gradients
            // dw = (1/m) * X^T * errors
            // db = (1/m) * sum(errors)

            let mut weight_gradients = vec![0.0; X.cols];
            for j in 0..X.cols {
                let mut grad = 0.0;
                for i in 0..X.rows {
                    grad += X[(i, j)] * errors.data[i];
                }
                weight_gradients[j] = grad / m;
            }

            let bias_gradient: f64 = errors.data.iter().sum::<f64>() / m;

            // Update parameters
            for j in 0..self.weights.data.len() {
                self.weights.data[j] -= self.learning_rate * weight_gradients[j];
            }
            self.bias -= self.learning_rate * bias_gradient;
        }

        println!("\nTraining complete!");
        println!(
            "Final cost: {:.6}",
            self.training_history.last().unwrap_or(&0.0)
        );
        println!("Weights: {:?}", self.weights.data);
        println!("Bias: {:.6}", self.bias);
    }

    /// Train with early stopping based on cost threshold
    pub fn fit_until_converged(
        &mut self,
        X: &Matrix<f64>,
        y: &Vector<f64>,
        max_iterations: usize,
        tolerance: f64,
    ) {
        assert_eq!(X.rows, y.data.len(), "X rows must match y length");

        if self.weights.data.is_empty() {
            self.initialize(X.cols);
        }

        let m = X.rows as f64;

        for iter in 0..max_iterations {
            let predictions = self.predict(X);
            let current_cost = Self::cost(&predictions, y);
            self.training_history.push(current_cost);

            if iter % 100 == 0 {
                println!("Iteration {}: Cost = {:.6}", iter, current_cost);
            }

            // Check for convergence
            if current_cost < tolerance {
                println!(
                    "\nConverged at iteration {} with cost {:.6}",
                    iter, current_cost
                );
                break;
            }

            // Compute gradients and update
            let errors = predictions - y.clone();

            let mut weight_gradients = vec![0.0; X.cols];
            for j in 0..X.cols {
                let mut grad = 0.0;
                for i in 0..X.rows {
                    grad += X[(i, j)] * errors.data[i];
                }
                weight_gradients[j] = grad / m;
            }

            let bias_gradient: f64 = errors.data.iter().sum::<f64>() / m;

            for j in 0..self.weights.data.len() {
                self.weights.data[j] -= self.learning_rate * weight_gradients[j];
            }
            self.bias -= self.learning_rate * bias_gradient;
        }
    }

    /// Lazily traings the model by Iterateing over lines in `filepath`
    pub fn fit_from_file<P: AsRef<Path>>(
        &self,
        filepath: P,
        headers: bool,
        training_columns: Vec<usize>,
        target_column: usize,
    ) {
        if let Ok(lines) = read(filepath) {
            for line in lines.map_while(Result::ok) {
                println!("{}", line);
            }
        }
    }
}

// Implement loader traits for LinearRegressor
impl loader::Trainable for LinearRegressor {
    type Error = String;

    fn fit(&mut self, features: &Matrix<f64>, targets: &[f64]) -> Result<(), Self::Error> {
        // Convert Vec<f64> targets to Vector<f64>
        let y = Vector {
            data: targets.to_vec(),
        };

        // Use the existing fit method with a default number of iterations
        // Note: This uses 1000 iterations as a reasonable default
        self.fit(features, &y, 1000);

        Ok(())
    }

    fn loss_history(&self) -> Vec<f64> {
        self.training_history.clone()
    }
}

impl loader::Predictable for LinearRegressor {
    type Error = String;

    fn predict(&self, features: &Matrix<f64>) -> Result<Vec<f64>, Self::Error> {
        if self.weights.data.is_empty() {
            return Err("Model not trained: weights are empty".to_string());
        }

        if features.cols != self.weights.data.len() {
            return Err(format!(
                "Feature dimension mismatch: expected {}, got {}",
                self.weights.data.len(),
                features.cols
            ));
        }

        // Use the existing predict method and convert Vector to Vec
        let predictions = self.predict(features);
        Ok(predictions.data)
    }
}

// Auto-implement Model trait
impl loader::Model for LinearRegressor {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        let mut model = LinearRegressor::new(0.01);
        assert_eq!(model.learning_rate, 0.01);
        assert_eq!(model.weights.data.len(), 0);
        assert_eq!(model.bias, 0.0);

        model.initialize(3);
        assert_eq!(model.weights.data.len(), 3);
        assert_eq!(model.weights.data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_predict_single() {
        let mut model = LinearRegressor::new(0.01);
        model.weights = Vector {
            data: vec![2.0, 3.0],
        };
        model.bias = 1.0;

        let x = Vector {
            data: vec![4.0, 5.0],
        };
        // prediction = 2*4 + 3*5 + 1 = 8 + 15 + 1 = 24
        let pred = model.predict_single(&x);
        assert_eq!(pred, 24.0);
    }

    #[test]
    fn test_predict_batch() {
        let mut model = LinearRegressor::new(0.01);
        model.weights = Vector {
            data: vec![2.0, 3.0],
        };
        model.bias = 1.0;

        // X = [[1, 2], [3, 4]]
        let X = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let predictions = model.predict(&X);

        // Row 0: 2*1 + 3*2 + 1 = 2 + 6 + 1 = 9
        // Row 1: 2*3 + 3*4 + 1 = 6 + 12 + 1 = 19
        assert_eq!(predictions.data, vec![9.0, 19.0]);
    }

    #[test]
    fn test_cost_calculation() {
        let predictions = Vector {
            data: vec![2.0, 4.0, 6.0],
        };
        let targets = Vector {
            data: vec![1.0, 3.0, 5.0],
        };

        // MSE = (1/2m) * sum((pred - target)^2)
        // errors = [1, 1, 1]
        // sum of squares = 3
        // MSE = 3 / (2*3) = 0.5
        let cost = LinearRegressor::cost(&predictions, &targets);
        assert_eq!(cost, 0.5);
    }

    #[test]
    fn test_perfect_fit() {
        // Test on perfectly linear data: y = 2x + 1
        let mut model = LinearRegressor::new(0.01);

        // Training data
        let X = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], 5, 1).unwrap();
        let y = Vector {
            data: vec![3.0, 5.0, 7.0, 9.0, 11.0],
        };

        // Train
        model.fit(&X, &y, 1000);

        // Check that learned parameters are close to true values
        // weight should be close to 2.0
        // bias should be close to 1.0
        assert!((model.weights.data[0] - 2.0).abs() < 0.1);
        assert!((model.bias - 1.0).abs() < 0.1);

        // Final cost should be very low
        let final_cost = model.training_history.last().unwrap();
        assert!(final_cost < &0.01);
    }

    #[test]
    fn test_multivariate_regression() {
        // Test with multiple features: y = 2x1 + 3x2 + 1
        let mut model = LinearRegressor::new(0.01);

        // Training data with 2 independent features
        let X = Matrix::from_vec(
            vec![
                1.0, 2.0, // sample 1: x1=1, x2=2
                2.0, 1.0, // sample 2: x1=2, x2=1
                3.0, 4.0, // sample 3: x1=3, x2=4
                4.0, 3.0, // sample 4: x1=4, x2=3
                2.0, 3.0, // sample 5: x1=2, x2=3
                3.0, 2.0, // sample 6: x1=3, x2=2
            ],
            6,
            2,
        )
        .unwrap();

        // y = 2*x1 + 3*x2 + 1
        let y = Vector {
            data: vec![
                2.0 * 1.0 + 3.0 * 2.0 + 1.0, // 9
                2.0 * 2.0 + 3.0 * 1.0 + 1.0, // 8
                2.0 * 3.0 + 3.0 * 4.0 + 1.0, // 19
                2.0 * 4.0 + 3.0 * 3.0 + 1.0, // 18
                2.0 * 2.0 + 3.0 * 3.0 + 1.0, // 14
                2.0 * 3.0 + 3.0 * 2.0 + 1.0, // 13
            ],
        };

        model.fit(&X, &y, 3000);

        // Check learned parameters (more tolerance due to numerical precision)
        assert!((model.weights.data[0] - 2.0).abs() < 0.3);
        assert!((model.weights.data[1] - 3.0).abs() < 0.3);
        assert!((model.bias - 1.0).abs() < 0.3);
    }

    #[test]
    fn test_training_history() {
        let mut model = LinearRegressor::new(0.01);
        let X = Matrix::from_vec(vec![1.0, 2.0, 3.0], 3, 1).unwrap();
        let y = Vector {
            data: vec![2.0, 4.0, 6.0],
        };

        model.fit(&X, &y, 100);

        // Training history should have 100 entries
        assert_eq!(model.training_history.len(), 100);

        // Cost should be decreasing
        let first_cost = model.training_history[0];
        let last_cost = model.training_history[99];
        assert!(last_cost < first_cost);
    }

    #[test]
    fn test_convergence() {
        let mut model = LinearRegressor::new(0.1);
        let X = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 4, 1).unwrap();
        let y = Vector {
            data: vec![2.0, 4.0, 6.0, 8.0],
        };

        model.fit_until_converged(&X, &y, 10000, 0.01);

        // Should have converged (cost < tolerance)
        let final_cost = model.training_history.last().unwrap();
        assert!(final_cost < &0.01);

        // Should not have used all iterations
        assert!(model.training_history.len() < 10000);
    }
}
