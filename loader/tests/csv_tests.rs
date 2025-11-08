use loader::{CsvDataset, CsvError};

#[test]
fn test_valid_csv_parsing() {
    let csv = "x1,x2,y\n1.0,2.0,3.0\n4.0,5.0,6.0";
    let dataset = CsvDataset::from_csv(csv, "y").unwrap();

    assert_eq!(dataset.num_samples, 2);
    assert_eq!(dataset.features.rows, 2);
    assert_eq!(dataset.features.cols, 2);
    assert_eq!(dataset.targets, vec![3.0, 6.0]);
    assert_eq!(dataset.feature_names, vec!["x1", "x2"]);
}

#[test]
fn test_single_feature_csv() {
    let csv = "feature,target\n1.5,10.0\n2.5,20.0\n3.5,30.0";
    let dataset = CsvDataset::from_csv(csv, "target").unwrap();

    assert_eq!(dataset.num_samples, 3);
    assert_eq!(dataset.features.rows, 3);
    assert_eq!(dataset.features.cols, 1);
    assert_eq!(dataset.targets, vec![10.0, 20.0, 30.0]);
    assert_eq!(dataset.feature_names, vec!["feature"]);
}

#[test]
fn test_multiple_features() {
    let csv = "f1,f2,f3,label\n1.0,2.0,3.0,100.0\n4.0,5.0,6.0,200.0";
    let dataset = CsvDataset::from_csv(csv, "label").unwrap();

    assert_eq!(dataset.num_samples, 2);
    assert_eq!(dataset.features.rows, 2);
    assert_eq!(dataset.features.cols, 3);
    assert_eq!(dataset.targets, vec![100.0, 200.0]);
    assert_eq!(dataset.feature_names, vec!["f1", "f2", "f3"]);

    // Verify feature matrix values
    assert_eq!(*dataset.features.get(0, 0).unwrap(), 1.0);
    assert_eq!(*dataset.features.get(0, 1).unwrap(), 2.0);
    assert_eq!(*dataset.features.get(0, 2).unwrap(), 3.0);
    assert_eq!(*dataset.features.get(1, 0).unwrap(), 4.0);
    assert_eq!(*dataset.features.get(1, 1).unwrap(), 5.0);
    assert_eq!(*dataset.features.get(1, 2).unwrap(), 6.0);
}

#[test]
fn test_target_as_first_column() {
    let csv = "target,f1,f2\n10.0,1.0,2.0\n20.0,3.0,4.0";
    let dataset = CsvDataset::from_csv(csv, "target").unwrap();

    assert_eq!(dataset.num_samples, 2);
    assert_eq!(dataset.features.cols, 2);
    assert_eq!(dataset.targets, vec![10.0, 20.0]);
    assert_eq!(dataset.feature_names, vec!["f1", "f2"]);
}

#[test]
fn test_target_as_middle_column() {
    let csv = "f1,target,f2\n1.0,10.0,2.0\n3.0,20.0,4.0";
    let dataset = CsvDataset::from_csv(csv, "target").unwrap();

    assert_eq!(dataset.num_samples, 2);
    assert_eq!(dataset.features.cols, 2);
    assert_eq!(dataset.targets, vec![10.0, 20.0]);
    assert_eq!(dataset.feature_names, vec!["f1", "f2"]);

    // Verify correct feature ordering (f1, then f2, skipping target)
    assert_eq!(*dataset.features.get(0, 0).unwrap(), 1.0);
    assert_eq!(*dataset.features.get(0, 1).unwrap(), 2.0);
}

#[test]
fn test_negative_and_decimal_values() {
    let csv = "x,y\n-1.5,2.3\n0.0,-5.7\n3.15,1.41";
    let dataset = CsvDataset::from_csv(csv, "y").unwrap();

    assert_eq!(dataset.num_samples, 3);
    assert_eq!(dataset.targets, vec![2.3, -5.7, 1.41]);
    assert_eq!(*dataset.features.get(0, 0).unwrap(), -1.5);
    assert_eq!(*dataset.features.get(1, 0).unwrap(), 0.0);
    assert_eq!(*dataset.features.get(2, 0).unwrap(), 3.15);
}

#[test]
fn test_missing_target_column() {
    let csv = "x1,x2\n1.0,2.0";
    let result = CsvDataset::from_csv(csv, "y");

    assert!(matches!(result, Err(CsvError::TargetNotFound(_))));
    if let Err(CsvError::TargetNotFound(col)) = result {
        assert_eq!(col, "y");
    }
}

#[test]
fn test_non_numeric_value_in_features() {
    let csv = "x,y\n1.0,2.0\nabc,3.0";
    let result = CsvDataset::from_csv(csv, "y");

    assert!(matches!(result, Err(CsvError::ParseError { .. })));
    if let Err(CsvError::ParseError {
        column,
        line,
        value,
    }) = result
    {
        assert_eq!(column, "x");
        assert_eq!(line, 3); // Header is line 1, first data row is line 2, second is line 3
        assert_eq!(value, "abc");
    }
}

#[test]
fn test_non_numeric_value_in_target() {
    let csv = "x,y\n1.0,abc\n2.0,3.0";
    let result = CsvDataset::from_csv(csv, "y");

    assert!(matches!(result, Err(CsvError::ParseError { .. })));
    if let Err(CsvError::ParseError {
        column,
        line,
        value,
    }) = result
    {
        assert_eq!(column, "y");
        assert_eq!(line, 2);
        assert_eq!(value, "abc");
    }
}

#[test]
fn test_infinite_value_rejection_in_features() {
    let csv = "x,y\ninf,2.0\n3.0,4.0";
    let result = CsvDataset::from_csv(csv, "y");

    assert!(matches!(result, Err(CsvError::InvalidValue { .. })));
    if let Err(CsvError::InvalidValue { column, line, .. }) = result {
        assert_eq!(column, "x");
        assert_eq!(line, 2);
    }
}

#[test]
fn test_infinite_value_rejection_in_target() {
    let csv = "x,y\n1.0,inf\n2.0,3.0";
    let result = CsvDataset::from_csv(csv, "y");

    assert!(matches!(result, Err(CsvError::InvalidValue { .. })));
    if let Err(CsvError::InvalidValue { column, line, .. }) = result {
        assert_eq!(column, "y");
        assert_eq!(line, 2);
    }
}

#[test]
fn test_nan_value_rejection() {
    let csv = "x,y\nNaN,2.0";
    let result = CsvDataset::from_csv(csv, "y");

    assert!(matches!(result, Err(CsvError::InvalidValue { .. })));
}

#[test]
fn test_empty_dataset() {
    let csv = "x,y";
    let result = CsvDataset::from_csv(csv, "y");

    assert!(matches!(result, Err(CsvError::EmptyDataset)));
}

#[test]
fn test_empty_dataset_with_newline() {
    let csv = "x,y\n";
    let result = CsvDataset::from_csv(csv, "y");

    assert!(matches!(result, Err(CsvError::EmptyDataset)));
}

#[test]
fn test_inconsistent_columns_too_few() {
    let csv = "x1,x2,y\n1.0,2.0,3.0\n4.0,5.0";
    let result = CsvDataset::from_csv(csv, "y");

    // CSV parser may automatically handle this - let's check what error we get
    match result {
        Err(CsvError::InconsistentColumns {
            expected,
            found,
            line,
        }) => {
            assert_eq!(expected, 3);
            assert_eq!(found, 2);
            assert_eq!(line, 3);
        }
        Err(CsvError::Csv(_)) => {
            // CSV crate may catch this earlier - that's also acceptable
        }
        Ok(_) => panic!("Expected an error for inconsistent columns"),
        Err(e) => panic!("Unexpected error type: {:?}", e),
    }
}

#[test]
fn test_inconsistent_columns_too_many() {
    let csv = "x1,x2,y\n1.0,2.0,3.0\n4.0,5.0,6.0,7.0";
    let result = CsvDataset::from_csv(csv, "y");

    // CSV parser may automatically handle this - let's check what error we get
    match result {
        Err(CsvError::InconsistentColumns {
            expected,
            found,
            line,
        }) => {
            assert_eq!(expected, 3);
            assert_eq!(found, 4);
            assert_eq!(line, 3);
        }
        Err(CsvError::Csv(_)) => {
            // CSV crate may catch this earlier - that's also acceptable
        }
        Ok(_) => panic!("Expected an error for inconsistent columns"),
        Err(e) => panic!("Unexpected error type: {:?}", e),
    }
}

#[test]
fn test_whitespace_handling() {
    // Test that column names with spaces can be matched
    // Note: CSV standard doesn't auto-trim unless configured
    let csv = "x,y\n1.0,2.0\n3.0,4.0";
    let dataset = CsvDataset::from_csv(csv, "y").unwrap();

    assert_eq!(dataset.num_samples, 2);
    assert_eq!(dataset.feature_names[0], "x");

    // If column names have trailing spaces, they must be matched exactly
    let csv_with_spaces = "x ,y \n1.0,2.0\n3.0,4.0";
    let result = CsvDataset::from_csv(csv_with_spaces, "y ");
    assert!(result.is_ok());
}

#[test]
fn test_large_values() {
    let csv = "x,y\n1e100,2e100\n3e-100,4e-100";
    let dataset = CsvDataset::from_csv(csv, "y").unwrap();

    assert_eq!(dataset.num_samples, 2);
    assert_eq!(dataset.targets[0], 2e100);
    assert_eq!(dataset.targets[1], 4e-100);
    assert_eq!(*dataset.features.get(0, 0).unwrap(), 1e100);
    assert_eq!(*dataset.features.get(1, 0).unwrap(), 3e-100);
}

#[test]
fn test_zero_values() {
    let csv = "x,y\n0.0,0.0\n-0.0,0.0";
    let dataset = CsvDataset::from_csv(csv, "y").unwrap();

    assert_eq!(dataset.num_samples, 2);
    assert_eq!(dataset.targets, vec![0.0, 0.0]);
}

#[test]
fn test_many_samples() {
    // Generate CSV with 100 samples
    let mut csv = String::from("x,y\n");
    for i in 0..100 {
        csv.push_str(&format!("{}.0,{}.0\n", i, i * 2));
    }

    let dataset = CsvDataset::from_csv(&csv, "y").unwrap();

    assert_eq!(dataset.num_samples, 100);
    assert_eq!(dataset.features.rows, 100);
    assert_eq!(dataset.targets.len(), 100);

    // Verify first and last samples
    assert_eq!(*dataset.features.get(0, 0).unwrap(), 0.0);
    assert_eq!(dataset.targets[0], 0.0);
    assert_eq!(*dataset.features.get(99, 0).unwrap(), 99.0);
    assert_eq!(dataset.targets[99], 198.0);
}

#[test]
fn test_error_display_messages() {
    // Test that error messages are informative
    let csv = "x,y\n1.0,abc";
    let result = CsvDataset::from_csv(csv, "y");

    assert!(matches!(result, Err(CsvError::ParseError { .. })));
    let err = result.unwrap_err();
    let message = format!("{}", err);
    assert!(message.contains("Line 2"));
    assert!(message.contains("column 'y'"));
    assert!(message.contains("abc"));
}

#[test]
fn test_csv_clone() {
    let csv = "x,y\n1.0,2.0\n3.0,4.0";
    let dataset1 = CsvDataset::from_csv(csv, "y").unwrap();
    let dataset2 = dataset1.clone();

    assert_eq!(dataset1.num_samples, dataset2.num_samples);
    assert_eq!(dataset1.targets, dataset2.targets);
    assert_eq!(dataset1.feature_names, dataset2.feature_names);
}
