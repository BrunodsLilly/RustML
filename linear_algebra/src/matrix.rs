use crate::vectors::Vector;
use std::ops::{Add, Index, IndexMut, Mul, Sub};

/// A matrix stored in row-major order
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    pub data: Vec<T>,
    pub rows: usize,
    pub cols: usize,
}

impl<T> Matrix<T>
where
    T: Copy + Default,
{
    /// Create a new matrix with specified dimensions, filled with default values
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![T::default(); rows * cols],
            rows,
            cols,
        }
    }

    /// Create a matrix from a flat vector and dimensions
    pub fn from_vec(data: Vec<T>, rows: usize, cols: usize) -> Result<Self, String> {
        if data.len() != rows * cols {
            return Err(format!(
                "Data length {} doesn't match dimensions {}x{}",
                data.len(),
                rows,
                cols
            ));
        }
        Ok(Matrix { data, rows, cols })
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            Some(&self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Get mutable element at (row, col)
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.rows && col < self.cols {
            Some(&mut self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Get the shape of the matrix as (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Transpose the matrix
    pub fn transpose(&self) -> Matrix<T> {
        let mut result = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                if let Some(&val) = self.get(i, j) {
                    if let Some(dest) = result.get_mut(j, i) {
                        *dest = val;
                    }
                }
            }
        }
        result
    }

    /// Get a specific row as a Vector
    pub fn row(&self, row: usize) -> Option<Vector<T>> {
        if row >= self.rows {
            return None;
        }
        let start = row * self.cols;
        let end = start + self.cols;
        Some(Vector {
            data: self.data[start..end].to_vec(),
        })
    }

    /// Get a specific column as a Vector
    pub fn col(&self, col: usize) -> Option<Vector<T>> {
        if col >= self.cols {
            return None;
        }
        let mut data = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            if let Some(&val) = self.get(i, col) {
                data.push(val);
            }
        }
        Some(Vector { data })
    }
}

// Matrix creation helpers for numeric types
impl<T> Matrix<T>
where
    T: Copy + Default + From<i32>,
{
    /// Create a matrix filled with zeros
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix::new(rows, cols)
    }

    /// Create a matrix filled with ones
    pub fn ones(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![T::from(1); rows * cols],
            rows,
            cols,
        }
    }

    /// Create an identity matrix (square)
    pub fn identity(size: usize) -> Self {
        let mut matrix = Matrix::zeros(size, size);
        for i in 0..size {
            if let Some(elem) = matrix.get_mut(i, i) {
                *elem = T::from(1);
            }
        }
        matrix
    }
}

// Indexing with (row, col) tuple
impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        &self.data[row * self.cols + col]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        &mut self.data[row * self.cols + col]
    }
}

// Matrix addition
impl<T> Add for Matrix<T>
where
    T: Copy + Default + Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Matrices must have the same shape for addition"
        );
        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

// Matrix subtraction
impl<T> Sub for Matrix<T>
where
    T: Copy + Default + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Matrices must have the same shape for subtraction"
        );
        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

// Matrix-Vector multiplication (Mv = v')
impl<T> Mul<Vector<T>> for Matrix<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    type Output = Vector<T>;

    fn mul(self, vec: Vector<T>) -> Self::Output {
        assert_eq!(
            self.cols,
            vec.data.len(),
            "Matrix columns must match vector length"
        );

        let mut result = vec![T::default(); self.rows];
        for i in 0..self.rows {
            let mut sum = T::default();
            for j in 0..self.cols {
                sum = sum + (self[(i, j)] * vec.data[j]);
            }
            result[i] = sum;
        }
        Vector { data: result }
    }
}

// Matrix-Matrix multiplication
impl<T> Mul for Matrix<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        assert_eq!(
            self.cols, other.rows,
            "Matrix dimensions incompatible for multiplication"
        );

        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::default();
                for k in 0..self.cols {
                    sum = sum + (self[(i, k)] * other[(k, j)]);
                }
                result[(i, j)] = sum;
            }
        }
        result
    }
}

// Scalar multiplication
impl<T> Mul<T> for Matrix<T>
where
    T: Copy + Default + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self::Output {
        let data: Vec<T> = self.data.iter().map(|&x| x * scalar).collect();
        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

/// Macro for creating matrices easily
#[macro_export]
macro_rules! matrix {
    ( $( [ $( $x:expr ),* ] ),* ) => {
        {
            let mut rows = Vec::new();
            $(
                let row = vec![$($x),*];
                rows.push(row);
            )*

            let num_rows = rows.len();
            let num_cols = if num_rows > 0 { rows[0].len() } else { 0 };

            let mut data = Vec::with_capacity(num_rows * num_cols);
            for row in rows {
                assert_eq!(row.len(), num_cols, "All rows must have the same length");
                data.extend(row);
            }

            Matrix {
                data,
                rows: num_rows,
                cols: num_cols,
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m: Matrix<i32> = Matrix::new(2, 3);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.data.len(), 6);
    }

    #[test]
    fn test_matrix_from_vec() {
        let m = Matrix::from_vec(vec![1, 2, 3, 4, 5, 6], 2, 3).unwrap();
        assert_eq!(m[(0, 0)], 1);
        assert_eq!(m[(0, 1)], 2);
        assert_eq!(m[(1, 2)], 6);
    }

    #[test]
    fn test_matrix_zeros_ones() {
        let zeros: Matrix<i32> = Matrix::zeros(2, 2);
        assert_eq!(zeros[(0, 0)], 0);

        let ones: Matrix<i32> = Matrix::ones(2, 2);
        assert_eq!(ones[(0, 0)], 1);
        assert_eq!(ones[(1, 1)], 1);
    }

    #[test]
    fn test_matrix_identity() {
        let id: Matrix<i32> = Matrix::identity(3);
        assert_eq!(id[(0, 0)], 1);
        assert_eq!(id[(1, 1)], 1);
        assert_eq!(id[(2, 2)], 1);
        assert_eq!(id[(0, 1)], 0);
        assert_eq!(id[(1, 0)], 0);
    }

    #[test]
    fn test_matrix_indexing() {
        let mut m = Matrix::from_vec(vec![1, 2, 3, 4], 2, 2).unwrap();
        assert_eq!(m[(0, 0)], 1);
        assert_eq!(m[(1, 1)], 4);

        m[(0, 0)] = 10;
        assert_eq!(m[(0, 0)], 10);
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::from_vec(vec![1, 2, 3, 4, 5, 6], 2, 3).unwrap();
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t[(0, 0)], 1);
        assert_eq!(t[(1, 0)], 2);
        assert_eq!(t[(2, 1)], 6);
    }

    #[test]
    fn test_matrix_addition() {
        let m1 = Matrix::from_vec(vec![1, 2, 3, 4], 2, 2).unwrap();
        let m2 = Matrix::from_vec(vec![5, 6, 7, 8], 2, 2).unwrap();
        let result = m1 + m2;
        assert_eq!(result[(0, 0)], 6);
        assert_eq!(result[(1, 1)], 12);
    }

    #[test]
    fn test_matrix_vector_mul() {
        let m = Matrix::from_vec(vec![1, 2, 3, 4, 5, 6], 2, 3).unwrap();
        let v = Vector {
            data: vec![1, 2, 3],
        };
        let result = m * v;
        // [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
        assert_eq!(result.data, vec![14, 32]);
    }

    #[test]
    fn test_matrix_matrix_mul() {
        let m1 = Matrix::from_vec(vec![1, 2, 3, 4], 2, 2).unwrap();
        let m2 = Matrix::from_vec(vec![5, 6, 7, 8], 2, 2).unwrap();
        let result = m1 * m2;
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        assert_eq!(result[(0, 0)], 19);
        assert_eq!(result[(0, 1)], 22);
        assert_eq!(result[(1, 0)], 43);
        assert_eq!(result[(1, 1)], 50);
    }

    #[test]
    fn test_matrix_scalar_mul() {
        let m = Matrix::from_vec(vec![1, 2, 3, 4], 2, 2).unwrap();
        let result = m * 2;
        assert_eq!(result[(0, 0)], 2);
        assert_eq!(result[(1, 1)], 8);
    }

    #[test]
    fn test_matrix_macro() {
        let m: Matrix<i32> = matrix![[1, 2, 3], [4, 5, 6]];
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m[(0, 0)], 1);
        assert_eq!(m[(1, 2)], 6);
    }

    #[test]
    fn test_row_col_access() {
        let m = Matrix::from_vec(vec![1, 2, 3, 4, 5, 6], 2, 3).unwrap();

        let row0 = m.row(0).unwrap();
        assert_eq!(row0.data, vec![1, 2, 3]);

        let col1 = m.col(1).unwrap();
        assert_eq!(col1.data, vec![2, 5]);
    }
}

// Implement ml_traits::Data for Matrix<f64>
impl ml_traits::Data<f64> for Matrix<f64> {
    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row < self.rows && col < self.cols {
            Some(self.data[row * self.cols + col])
        } else {
            None
        }
    }

    fn n_samples(&self) -> usize {
        self.rows
    }

    fn n_features(&self) -> usize {
        self.cols
    }
}
