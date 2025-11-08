//! Loss Function Library for Optimizer Visualization
//!
//! This module provides a collection of classic optimization test functions.
//! Each function has interesting characteristics that reveal optimizer behavior:
//! - Rosenbrock: Narrow curved valley (tests valley navigation)
//! - Beale: Multiple local minima (tests exploration)
//! - Himmelblau: Four global minima (tests multi-modal optimization)
//! - Saddle Point: Tests escape from saddle points
//! - Rastrigin: Highly multi-modal (tests global vs local search)
//!
//! All functions are optimized for WASM performance with SIMD-friendly operations.

use std::f64::consts::PI;

/// A test function for optimization with known properties
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LossFunction {
    /// Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
    /// Global minimum at (1, 1), f(1,1) = 0
    /// Famous narrow curved valley - tests momentum and adaptive learning rates
    Rosenbrock,

    /// Beale: f(x,y) = (1.5-x+xy)² + (2.25-x+xy²)² + (2.625-x+xy³)²
    /// Global minimum at (3, 0.5), f(3, 0.5) = 0
    /// Tests handling of narrow valleys with steep walls
    Beale,

    /// Himmelblau: f(x,y) = (x²+y-11)² + (x+y²-7)²
    /// Four global minima - tests multi-modal optimization
    Himmelblau,

    /// Saddle Point: f(x,y) = x² - y²
    /// Classic saddle point at origin - tests escape capability
    Saddle,

    /// Rastrigin: f(x,y) = 20 + x²-10cos(2πx) + y²-10cos(2πy)
    /// Highly multi-modal with many local minima
    /// Global minimum at (0, 0)
    Rastrigin,

    /// Simple Quadratic: f(x,y) = x² + y²
    /// Simplest case - perfect bowl shape
    /// Good for debugging and baseline testing
    Quadratic,
}

impl LossFunction {
    /// Evaluate the loss function at point (x, y)
    #[inline]
    pub fn evaluate(&self, x: f64, y: f64) -> f64 {
        match self {
            Self::Rosenbrock => {
                let a = 1.0 - x;
                let b = y - x * x;
                a * a + 100.0 * b * b
            }

            Self::Beale => {
                let t1 = 1.5 - x + x * y;
                let t2 = 2.25 - x + x * y * y;
                let t3 = 2.625 - x + x * y * y * y;
                t1 * t1 + t2 * t2 + t3 * t3
            }

            Self::Himmelblau => {
                let t1 = x * x + y - 11.0;
                let t2 = x + y * y - 7.0;
                t1 * t1 + t2 * t2
            }

            Self::Saddle => x * x - y * y,

            Self::Rastrigin => {
                let a = 10.0;
                2.0 * a + (x * x - a * (2.0 * PI * x).cos()) + (y * y - a * (2.0 * PI * y).cos())
            }

            Self::Quadratic => x * x + y * y,
        }
    }

    /// Compute the gradient ∇f at point (x, y)
    /// Returns (∂f/∂x, ∂f/∂y)
    #[inline]
    pub fn gradient(&self, x: f64, y: f64) -> (f64, f64) {
        match self {
            Self::Rosenbrock => {
                let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
                let dy = 200.0 * (y - x * x);
                (dx, dy)
            }

            Self::Beale => {
                let t1 = 1.5 - x + x * y;
                let t2 = 2.25 - x + x * y * y;
                let t3 = 2.625 - x + x * y * y * y;

                let dx = 2.0 * t1 * (-1.0 + y)
                    + 2.0 * t2 * (-1.0 + y * y)
                    + 2.0 * t3 * (-1.0 + y * y * y);
                let dy = 2.0 * t1 * x + 2.0 * t2 * (2.0 * x * y) + 2.0 * t3 * (3.0 * x * y * y);
                (dx, dy)
            }

            Self::Himmelblau => {
                let t1 = x * x + y - 11.0;
                let t2 = x + y * y - 7.0;

                let dx = 2.0 * t1 * (2.0 * x) + 2.0 * t2;
                let dy = 2.0 * t1 + 2.0 * t2 * (2.0 * y);
                (dx, dy)
            }

            Self::Saddle => (2.0 * x, -2.0 * y),

            Self::Rastrigin => {
                let a = 10.0;
                let dx = 2.0 * x + 2.0 * PI * a * (2.0 * PI * x).sin();
                let dy = 2.0 * y + 2.0 * PI * a * (2.0 * PI * y).sin();
                (dx, dy)
            }

            Self::Quadratic => (2.0 * x, 2.0 * y),
        }
    }

    /// Get the recommended search bounds for visualization
    /// Returns ((x_min, x_max), (y_min, y_max))
    pub fn bounds(&self) -> ((f64, f64), (f64, f64)) {
        match self {
            Self::Rosenbrock => ((-2.0, 2.0), (-1.0, 3.0)),
            Self::Beale => ((-4.5, 4.5), (-4.5, 4.5)),
            Self::Himmelblau => ((-5.0, 5.0), (-5.0, 5.0)),
            Self::Saddle => ((-2.0, 2.0), (-2.0, 2.0)),
            Self::Rastrigin => ((-5.12, 5.12), (-5.12, 5.12)),
            Self::Quadratic => ((-3.0, 3.0), (-3.0, 3.0)),
        }
    }

    /// Get the global minimum point(s)
    /// Returns a vector of (x, y, f(x,y)) tuples
    pub fn global_minima(&self) -> Vec<(f64, f64, f64)> {
        match self {
            Self::Rosenbrock => vec![(1.0, 1.0, 0.0)],
            Self::Beale => vec![(3.0, 0.5, 0.0)],
            Self::Himmelblau => vec![
                (3.0, 2.0, 0.0),
                (-2.805118, 3.131312, 0.0),
                (-3.779310, -3.283186, 0.0),
                (3.584428, -1.848126, 0.0),
            ],
            Self::Saddle => vec![(0.0, 0.0, 0.0)], // Saddle point, not minimum
            Self::Rastrigin => vec![(0.0, 0.0, 0.0)],
            Self::Quadratic => vec![(0.0, 0.0, 0.0)],
        }
    }

    /// Get a good starting point for optimization
    pub fn starting_point(&self) -> (f64, f64) {
        match self {
            Self::Rosenbrock => (-1.0, 1.0),
            Self::Beale => (1.0, 1.0),
            Self::Himmelblau => (0.0, 0.0),
            Self::Saddle => (1.0, 1.0),
            Self::Rastrigin => (2.0, 2.0),
            Self::Quadratic => (2.0, 2.0),
        }
    }

    /// Get a human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Rosenbrock => "Rosenbrock",
            Self::Beale => "Beale",
            Self::Himmelblau => "Himmelblau",
            Self::Saddle => "Saddle Point",
            Self::Rastrigin => "Rastrigin",
            Self::Quadratic => "Quadratic Bowl",
        }
    }

    /// Get a description of what this function tests
    pub fn description(&self) -> &'static str {
        match self {
            Self::Rosenbrock => "Narrow curved valley - tests momentum & adaptive rates",
            Self::Beale => "Steep valleys with flat regions - tests robustness",
            Self::Himmelblau => "Four global minima - tests multi-modal search",
            Self::Saddle => "Saddle point at origin - tests escape capability",
            Self::Rastrigin => "Highly multi-modal - tests global optimization",
            Self::Quadratic => "Perfect bowl - baseline test case",
        }
    }

    /// Get the difficulty level (1-5)
    pub fn difficulty(&self) -> u8 {
        match self {
            Self::Quadratic => 1,
            Self::Saddle => 2,
            Self::Rosenbrock => 3,
            Self::Himmelblau => 4,
            Self::Beale => 4,
            Self::Rastrigin => 5,
        }
    }

    /// Generate a 2D heatmap for visualization
    /// Returns a grid of loss values
    pub fn generate_heatmap(&self, resolution: usize) -> Vec<Vec<f64>> {
        let ((x_min, x_max), (y_min, y_max)) = self.bounds();
        let mut grid = vec![vec![0.0; resolution]; resolution];

        // Standard grid indexing: grid[row][col] where row = y-axis, col = x-axis
        for row in 0..resolution {
            let y = y_min + (y_max - y_min) * (row as f64 / (resolution - 1) as f64);
            for col in 0..resolution {
                let x = x_min + (x_max - x_min) * (col as f64 / (resolution - 1) as f64);
                grid[row][col] = self.evaluate(x, y);
            }
        }

        grid
    }

    /// Get all available loss functions
    pub fn all() -> Vec<Self> {
        vec![
            Self::Quadratic,
            Self::Rosenbrock,
            Self::Saddle,
            Self::Beale,
            Self::Himmelblau,
            Self::Rastrigin,
        ]
    }
}

impl Default for LossFunction {
    fn default() -> Self {
        Self::Rosenbrock // Most pedagogically interesting
    }
}

/// Pre-computed heatmap for WASM performance
/// This struct caches the heatmap to avoid recomputation
#[derive(Clone, PartialEq)]
pub struct HeatmapCache {
    pub function: LossFunction,
    pub resolution: usize,
    pub grid: Vec<Vec<f64>>,
    pub min_value: f64,
    pub max_value: f64,
}

impl HeatmapCache {
    /// Create a new heatmap cache
    pub fn new(function: LossFunction, resolution: usize) -> Self {
        let grid = function.generate_heatmap(resolution);

        // Find min and max for normalization
        let mut min_value = f64::INFINITY;
        let mut max_value = f64::NEG_INFINITY;

        for row in &grid {
            for &val in row {
                if val.is_finite() {
                    min_value = min_value.min(val);
                    max_value = max_value.max(val);
                }
            }
        }

        Self {
            function,
            resolution,
            grid,
            min_value,
            max_value,
        }
    }

    /// Get normalized value (0.0 to 1.0) for color mapping
    ///
    /// # Arguments
    /// * `row` - Row index (y-axis)
    /// * `col` - Column index (x-axis)
    pub fn normalized_value(&self, row: usize, col: usize) -> f64 {
        let val = self.grid[row][col];
        if !val.is_finite() {
            return 1.0;
        }

        // Use log scale for better visualization of wide ranges
        let log_val = (val - self.min_value + 1.0).ln();
        let log_range = (self.max_value - self.min_value + 1.0).ln();

        (log_val / log_range).clamp(0.0, 1.0)
    }

    /// Map normalized value to color (HSL)
    /// Returns (hue, saturation, lightness) in [0, 1]
    pub fn value_to_color(&self, normalized: f64) -> (f64, f64, f64) {
        // Beautiful gradient from blue (low) -> cyan -> green -> yellow -> red (high)
        let hue = (1.0 - normalized) * 240.0; // 240° (blue) to 0° (red)
        let saturation = 0.8;
        let lightness = 0.3 + normalized * 0.4; // Darker for low, lighter for high

        (hue / 360.0, saturation, lightness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rosenbrock_minimum() {
        let f = LossFunction::Rosenbrock;
        let (x, y, expected) = f.global_minima()[0];
        let actual = f.evaluate(x, y);
        assert!(
            (actual - expected).abs() < 1e-10,
            "Rosenbrock minimum should be 0"
        );
    }

    #[test]
    fn test_gradients_numerical() {
        let f = LossFunction::Rosenbrock;
        let (x, y) = (0.5, 0.3);
        let (dx, dy) = f.gradient(x, y);

        // Numerical gradient check
        let eps = 1e-7;
        let dx_numerical = (f.evaluate(x + eps, y) - f.evaluate(x - eps, y)) / (2.0 * eps);
        let dy_numerical = (f.evaluate(x, y + eps) - f.evaluate(x, y - eps)) / (2.0 * eps);

        assert!((dx - dx_numerical).abs() < 1e-5, "dx gradient mismatch");
        assert!((dy - dy_numerical).abs() < 1e-5, "dy gradient mismatch");
    }

    #[test]
    fn test_all_functions_evaluate() {
        for func in LossFunction::all() {
            let (x, y) = func.starting_point();
            let loss = func.evaluate(x, y);
            assert!(
                loss.is_finite(),
                "{} should produce finite value",
                func.name()
            );

            let (dx, dy) = func.gradient(x, y);
            assert!(
                dx.is_finite() && dy.is_finite(),
                "{} gradients should be finite",
                func.name()
            );
        }
    }
}
