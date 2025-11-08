//! This crate implements data loading functionality

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

pub mod csv_loader;
pub use csv_loader::{CsvDataset, CsvError};

use linear_algebra::matrix::Matrix;

/// Trait for trainable ML models
pub trait Trainable {
    type Error;

    /// Train the model on features and targets
    fn fit(&mut self, features: &Matrix<f64>, targets: &[f64]) -> Result<(), Self::Error>;

    /// Get training loss history (optional, returns empty vec by default)
    fn loss_history(&self) -> Vec<f64> {
        vec![]
    }
}

/// Trait for models that can make predictions
pub trait Predictable {
    type Error;

    /// Predict targets for given features
    fn predict(&self, features: &Matrix<f64>) -> Result<Vec<f64>, Self::Error>;
}

/// Combined trait for full ML pipeline
pub trait Model: Trainable + Predictable {}

pub fn read<P>(filepath: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filepath)?;
    Ok(io::BufReader::new(file).lines())
}

pub fn head<P>(filepath: P, num_lines: usize) -> Result<String, String>
where
    P: AsRef<Path> + std::fmt::Debug,
{
    if let Ok(lines) = read(&filepath) {
        Ok(lines
            .map_while(Result::ok)
            .take(num_lines)
            .reduce(|acc, s| acc + "\n" + s.as_str())
            .unwrap())
    } else {
        Err(format!("Could not read {:?}", filepath))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = read("test_data.csv");
        let first_line = result.unwrap().next().unwrap().unwrap();
        assert_eq!(
            first_line,
            "Home,Price,SqFt,Bedrooms,Bathrooms,Offers,Brick,Neighborhood"
        );
    }
}
