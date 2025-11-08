use csv::ReaderBuilder;
use linear_algebra::matrix::Matrix;
use std::fmt;

/// Represents a parsed CSV dataset with features and targets separated.
#[derive(Debug, Clone)]
pub struct CsvDataset {
    /// Feature matrix (samples × features)
    pub features: Matrix<f64>,
    /// Target values (one per sample)
    pub targets: Vec<f64>,
    /// Names of feature columns
    pub feature_names: Vec<String>,
    /// Number of samples (rows) in the dataset
    pub num_samples: usize,
}

/// Error types for CSV parsing operations.
#[derive(Debug)]
pub enum CsvError {
    /// IO error during file reading
    Io(std::io::Error),
    /// CSV parsing error
    Csv(csv::Error),
    /// Matrix creation error
    Matrix(String),
    /// Target column not found in headers
    TargetNotFound(String),
    /// Inconsistent number of columns across rows
    InconsistentColumns {
        expected: usize,
        found: usize,
        line: usize,
    },
    /// Failed to parse numeric value
    ParseError {
        column: String,
        line: usize,
        value: String,
    },
    /// Invalid value (e.g., NaN, infinity)
    InvalidValue {
        column: String,
        line: usize,
        reason: String,
    },
    /// Dataset has no rows
    EmptyDataset,
}

impl fmt::Display for CsvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CsvError::Io(e) => write!(f, "IO error: {}", e),
            CsvError::Csv(e) => write!(f, "CSV parsing error: {}", e),
            CsvError::Matrix(e) => write!(f, "Matrix error: {}", e),
            CsvError::TargetNotFound(col) => {
                write!(f, "Target column '{}' not found in CSV headers", col)
            }
            CsvError::InconsistentColumns {
                expected,
                found,
                line,
            } => write!(
                f,
                "Line {}: expected {} columns, found {}",
                line, expected, found
            ),
            CsvError::ParseError {
                column,
                line,
                value,
            } => write!(
                f,
                "Line {}, column '{}': failed to parse '{}' as a number",
                line, column, value
            ),
            CsvError::InvalidValue {
                column,
                line,
                reason,
            } => {
                write!(f, "Line {}, column '{}': {}", line, column, reason)
            }
            CsvError::EmptyDataset => write!(f, "Dataset has no data rows"),
        }
    }
}

impl std::error::Error for CsvError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CsvError::Io(e) => Some(e),
            CsvError::Csv(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for CsvError {
    fn from(err: std::io::Error) -> Self {
        CsvError::Io(err)
    }
}

impl From<csv::Error> for CsvError {
    fn from(err: csv::Error) -> Self {
        CsvError::Csv(err)
    }
}

impl CsvDataset {
    /// Parse a CSV string into a dataset with automatic type inference and validation.
    ///
    /// # Arguments
    ///
    /// * `content` - CSV content as a string (must have headers)
    /// * `target_column` - Name of the column to use as the target variable
    ///
    /// # Returns
    ///
    /// * `Ok(CsvDataset)` - Successfully parsed dataset
    /// * `Err(CsvError)` - Parsing or validation error with line number and details
    ///
    /// # Examples
    ///
    /// ```
    /// use loader::CsvDataset;
    ///
    /// let csv = "feature1,feature2,target\n1.0,2.0,3.0\n4.0,5.0,6.0";
    /// let dataset = CsvDataset::from_csv(csv, "target").unwrap();
    ///
    /// assert_eq!(dataset.num_samples, 2);
    /// assert_eq!(dataset.features.rows, 2);
    /// assert_eq!(dataset.features.cols, 2);
    /// assert_eq!(dataset.targets, vec![3.0, 6.0]);
    /// ```
    pub fn from_csv(content: &str, target_column: &str) -> Result<Self, CsvError> {
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(content.as_bytes());

        // Get headers
        let headers = reader.headers()?.clone();

        // Validate target column exists
        let target_idx = headers
            .iter()
            .position(|h| h == target_column)
            .ok_or_else(|| CsvError::TargetNotFound(target_column.to_string()))?;

        // Parse records
        let mut features_data: Vec<Vec<f64>> = Vec::new();
        let mut targets_data: Vec<f64> = Vec::new();

        for (line_num, result) in reader.records().enumerate() {
            let record = result?;

            // Validate row length (line_num + 2 because: +1 for header row, +1 for 0-indexing)
            if record.len() != headers.len() {
                return Err(CsvError::InconsistentColumns {
                    expected: headers.len(),
                    found: record.len(),
                    line: line_num + 2,
                });
            }

            // Parse target
            let target = record[target_idx]
                .parse::<f64>()
                .map_err(|_| CsvError::ParseError {
                    column: target_column.to_string(),
                    line: line_num + 2,
                    value: record[target_idx].to_string(),
                })?;

            // Validate target is finite
            if !target.is_finite() {
                return Err(CsvError::InvalidValue {
                    column: target_column.to_string(),
                    line: line_num + 2,
                    reason: "non-finite value (NaN or infinity)".to_string(),
                });
            }
            targets_data.push(target);

            // Parse features (all columns except target)
            let mut row_features = Vec::with_capacity(headers.len() - 1);
            for (col_idx, value) in record.iter().enumerate() {
                if col_idx == target_idx {
                    continue; // Skip target column
                }

                let parsed = value.parse::<f64>().map_err(|_| CsvError::ParseError {
                    column: headers[col_idx].to_string(),
                    line: line_num + 2,
                    value: value.to_string(),
                })?;

                if !parsed.is_finite() {
                    return Err(CsvError::InvalidValue {
                        column: headers[col_idx].to_string(),
                        line: line_num + 2,
                        reason: "non-finite value (NaN or infinity)".to_string(),
                    });
                }
                row_features.push(parsed);
            }
            features_data.push(row_features);
        }

        // Validate minimum rows
        if features_data.is_empty() {
            return Err(CsvError::EmptyDataset);
        }

        // Convert to Matrix
        let num_samples = features_data.len();
        let num_features = features_data[0].len();
        let flat_features: Vec<f64> = features_data.into_iter().flatten().collect();
        let features =
            Matrix::from_vec(flat_features, num_samples, num_features).map_err(CsvError::Matrix)?;

        // Get feature names (exclude target)
        let feature_names: Vec<String> = headers
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != target_idx)
            .map(|(_, h)| h.to_string())
            .collect();

        Ok(CsvDataset {
            features,
            targets: targets_data,
            feature_names,
            num_samples,
        })
    }
}
