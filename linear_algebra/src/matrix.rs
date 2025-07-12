use crate::vectors::Vector;

pub struct Matrix<T> {
    data: Vec<T>,
}


impl<T>  for Matrix<T>
    where
        T: Copy + Default + $trait<Output = T>,
    {
        type Output = Self;

        fn $method(self, other: Self) -> Self::Output {
        }
    }
