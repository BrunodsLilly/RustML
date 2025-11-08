//! Create and implement a Vector for linear algebra

use std::ops::{Add, Div, Mul, Sub};

/// A vector for linear algebra
#[derive(Debug, PartialEq, Clone)]
pub struct Vector<T> {
    /// A standard vector
    pub data: Vec<T>,
}

impl<T> Default for Vector<T> {
    fn default() -> Self {
        Vector { data: vec![] }
    }
}

macro_rules! impl_vector_op {
    ($trait_path:path, $trait:ident, $method:ident, $op:expr) => {
        impl<T> $trait_path for Vector<T>
        where
            T: Copy + Default + $trait<Output = T>,
        {
            type Output = Self;

            fn $method(self, other: Self) -> Self::Output {
                assert_eq!(
                    self.data.len(),
                    other.data.len(),
                    "Vectors must be equal in length"
                );

                let mut new_data = Vec::with_capacity(self.data.len());

                for (v1, v2) in self.data.iter().zip(other.data.iter()) {
                    new_data.push($op(*v1, *v2));
                }
                Vector { data: new_data }
            }
        }
    };
}

impl_vector_op!(std::ops::Add, Add, add, |a, b| a + b);
impl_vector_op!(std::ops::Sub, Sub, sub, |a, b| a - b);
impl_vector_op!(std::ops::Mul, Mul, mul, |a, b| a * b);
impl_vector_op!(std::ops::Div, Div, div, |a, b| a / b);

macro_rules! impl_vector_scalar_op {
    ($trait_path:path, $trait:ident, $method:ident, $op:expr) => {
        impl<T> $trait<T> for Vector<T>
        where
            T: Copy + Default + $trait<Output = T>,
        {
            type Output = Self;

            fn $method(self, rhs: T) -> Self::Output {
                let mut new_data = Vec::with_capacity(self.data.len());

                for v in self.data.iter() {
                    new_data.push($op(*v, rhs));
                }
                Vector { data: new_data }
            }
        }
    };
}

impl_vector_scalar_op!(std::ops::Add, Add, add, |element, scalar| element + scalar);
impl_vector_scalar_op!(std::ops::Sub, Sub, sub, |element, scalar| element - scalar);
impl_vector_scalar_op!(std::ops::Mul, Mul, mul, |element, scalar| element * scalar);
impl_vector_scalar_op!(std::ops::Div, Div, div, |element, scalar| element / scalar);

// implement dot product

impl<T> Vector<T> {
    pub fn dot(&self, other: &Self) -> T
    where
        T: Copy + Default + Mul<Output = T> + Add<Output = T>,
    {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "Vectors must be equal in length for dot product"
        );

        let mut sum = T::default();

        for (v1, v2) in self.data.iter().zip(other.data.iter()) {
            sum = sum + (*v1 * *v2);
        }
        sum
    }
}

fn dot<T>(v1: Vector<T>, v2: Vector<T>) -> T
where
    T: Copy + Default + Mul<Output = T> + Add<Output = T>,
{
    v1.dot(&v2)
}

/// Quick way to create linear algebra vectors
#[macro_export]
macro_rules! v {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            Vector { data: temp_vec }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create() {
        let result = v![1, 2, 3];
        assert_eq!(result.data, vec![1, 2, 3]);
    }
    #[test]
    fn test_add() {
        let result = v![1, 2, 3] + v![4, 5, 6];
        assert_eq!(result, v![5, 7, 9]);
    }
    #[test]
    fn test_sub() {
        let result = v![1, 2, 3] - v![2, 3, 1];
        assert_eq!(result.data, vec![-1, -1, 2]);
    }
    #[test]
    fn test_mul() {
        let result = v![1, 2, 3] * v![2, 3, 1];
        assert_eq!(result.data, vec![2, 6, 3]);
    }
    #[test]
    fn test_div_int() {
        let result = v![1, 2, 3] / v![2, 3, 1];
        assert_eq!(result.data, vec![0, 0, 3]);
    }
    #[test]
    fn test_div_float() {
        let result = v![1.0, 2.0, 3.0] / v![2.0, 3.0, 1.0];
        assert_eq!(result.data, vec![1.0 / 2.0, 2.0 / 3.0, 3.0 / 1.0]);
    }

    #[test]
    fn test_add_scalar() {
        let result = v![1, 2, 3] + 10;
        assert_eq!(result, v![11, 12, 13]);
    }
    #[test]
    fn test_sub_scalar() {
        let result = v![1, 2, 3] - 10;
        assert_eq!(result, v![-9, -8, -7]);
    }
    #[test]
    fn test_mul_scalar() {
        let result = v![1, 2, 3] * 10;
        assert_eq!(result, v![10, 20, 30])
    }
    #[test]
    fn test_div_int_scalar() {
        let result = v![1, 2, 3] / 10;
        assert_eq!(result, v![0, 0, 0]);
    }
    #[test]
    fn test_div_float_scalar() {
        let result = v![1.0, 2.0, 3.0] / 10.0;
        assert_eq!(result, v![1.0 / 10.0, 2.0 / 10.0, 3.0 / 10.0])
    }

    #[test]
    fn test_dot_product() {
        let result = v![1, 2, 3].dot(&v![2, 4, 5]);
        assert_eq!(result, 25);
        let result = dot(v![1, 2, 3], v![2, 4, 5]);
        assert_eq!(result, 25);
    }
}
