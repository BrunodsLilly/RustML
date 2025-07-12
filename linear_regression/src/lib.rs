use std::{
    ops::{Add, Mul},
    path::Path,
};

use linear_algebra::{matrix::Matrix, vectors::Vector};
use loader::read;

#[derive(Default, Debug)]
pub struct LinearRegressor<T> {
    weights: Vector<T>,
    bias: T,
}
impl<T> LinearRegressor<T>
where
    T: Default + Copy + Mul<Output = T> + Add<Output = T>,
{
    pub fn new() -> LinearRegressor<T> {
        let model: LinearRegressor<T> = Default::default();
        model
    }

    pub fn predict(&self, X: Matrix<T>) -> Vector<T> {
        todo!()
    }

    pub fn cost(&self, x: Vector<T>, target: Vector<T>) -> f64 {
        let predictions = self.predict(x);
        (predictions - target)
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

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
