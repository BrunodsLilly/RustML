//! This crate implements data loading functionality

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

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
