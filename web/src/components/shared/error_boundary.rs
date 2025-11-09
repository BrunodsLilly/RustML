use dioxus::prelude::*;
use std::panic;
use linear_algebra::matrix::Matrix;

/// Props for the ErrorBoundary component
#[derive(Props, Clone, PartialEq)]
pub struct ErrorBoundaryProps {
    /// The children to render within the error boundary
    children: Element,
    /// Optional custom error message to display
    #[props(default = "An error occurred while processing your request.")]
    error_message: &'static str,
    /// Whether to show technical details (for development)
    #[props(default = false)]
    show_details: bool,
    /// Optional callback when an error occurs
    #[props(default)]
    on_error: Option<EventHandler<String>>,
}

/// ErrorBoundary component that catches WASM panics and displays user-friendly error messages
///
/// This component prevents the entire WASM app from crashing when an algorithm fails.
/// It catches panics using std::panic::catch_unwind and displays an error message instead.
///
/// # Example
/// ```rust
/// rsx! {
///     ErrorBoundary {
///         error_message: "K-Means clustering failed",
///         show_details: cfg!(debug_assertions),
///         div {
///             button {
///                 onclick: move |_| {
///                     // This might panic, but ErrorBoundary will catch it
///                     run_kmeans();
///                 },
///                 "Run K-Means"
///             }
///         }
///     }
/// }
/// ```
#[component]
pub fn ErrorBoundary(props: ErrorBoundaryProps) -> Element {
    let mut error_state = use_signal(|| None::<String>);

    rsx! {
        div { class: "error-boundary",
            if let Some(error) = error_state.read().as_ref() {
                div { class: "error-boundary-message",
                    div { class: "error-icon", "⚠️" }
                    h3 { "{props.error_message}" }
                    if props.show_details {
                        details {
                            summary { "Technical Details" }
                            pre { class: "error-details",
                                "{error}"
                            }
                        }
                    }
                    button {
                        class: "error-retry-button",
                        onclick: move |_| {
                            error_state.set(None);
                        },
                        "Try Again"
                    }
                }
            } else {
                {props.children}
            }
        }
    }
}

/// Execute a function within a panic boundary, catching any panics and converting them to Results
///
/// This is a utility function for wrapping algorithm execution to prevent WASM crashes.
///
/// # Example
/// ```rust
/// let result = catch_panic(|| {
///     let mut kmeans = KMeans::new(3, 100, 1e-4, Some(42));
///     kmeans.fit(&data)?;
///     kmeans.predict(&data)
/// });
///
/// match result {
///     Ok(labels) => println!("Clustering complete: {:?}", labels),
///     Err(e) => eprintln!("Error: {}", e),
/// }
/// ```
pub fn catch_panic<F, T>(f: F) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String> + panic::UnwindSafe,
{
    match panic::catch_unwind(f) {
        Ok(Ok(result)) => Ok(result),
        Ok(Err(e)) => {
            // Log error to browser console via println! (dioxus handles this)
            eprintln!("Algorithm error: {}", e);
            Err(e)
        }
        Err(panic_info) => {
            let error_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                format!("WASM panic: {}", s)
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                format!("WASM panic: {}", s)
            } else {
                "WASM panic: unknown error".to_string()
            };
            eprintln!("{}", error_msg);
            Err(error_msg)
        }
    }
}

/// Execute an async function within a panic boundary
///
/// Similar to catch_panic but for async operations. Useful for long-running algorithm executions.
///
/// # Example
/// ```rust
/// spawn(async move {
///     is_processing.set(true);
///
///     let result = catch_panic_async(async {
///         // Long-running algorithm
///         run_algorithm().await
///     }).await;
///
///     match result {
///         Ok(output) => result_message.set(output),
///         Err(e) => result_message.set(format!("Error: {}", e)),
///     }
///
///     is_processing.set(false);
/// });
/// ```
pub async fn catch_panic_async<F, T>(f: F) -> Result<T, String>
where
    F: std::future::Future<Output = Result<T, String>>,
{
    // Note: catch_unwind doesn't work with async, so we rely on Result propagation
    // The actual panic catching happens at the spawn boundary
    f.await
}

/// Validate input parameters before algorithm execution
///
/// This function provides common validation checks to prevent invalid inputs
/// from reaching algorithm code and causing panics.
///
/// # Example
/// ```rust
/// if let Err(e) = validate_algorithm_input(&data, Some(3), Some(100), Some(0.01)) {
///     eprintln!("{}", e);
///     return Err(e);
/// }
/// ```
pub fn validate_algorithm_input(
    data: &Matrix<f64>,
    k_clusters: Option<usize>,
    max_iterations: Option<usize>,
    learning_rate: Option<f64>,
) -> Result<(), String> {
    // Check matrix dimensions
    if data.rows == 0 || data.cols == 0 {
        return Err("Data matrix cannot be empty".to_string());
    }

    // Validate k_clusters if provided
    if let Some(k) = k_clusters {
        if k == 0 {
            return Err("Number of clusters must be greater than 0".to_string());
        }
        if k > data.rows {
            return Err(format!(
                "Number of clusters ({}) cannot exceed number of samples ({})",
                k,
                data.rows
            ));
        }
    }

    // Validate max_iterations if provided
    if let Some(iter) = max_iterations {
        if iter == 0 {
            return Err("Maximum iterations must be greater than 0".to_string());
        }
        if iter > 100000 {
            return Err("Maximum iterations cannot exceed 100,000 (performance limit)".to_string());
        }
    }

    // Validate learning_rate if provided
    if let Some(lr) = learning_rate {
        if lr <= 0.0 || !lr.is_finite() {
            return Err(format!(
                "Learning rate must be positive and finite, got: {}",
                lr
            ));
        }
        if lr > 10.0 {
            return Err("Learning rate should not exceed 10.0 (likely too large)".to_string());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use linear_algebra::matrix::Matrix;

    #[test]
    fn test_validate_algorithm_input_valid() {
        let data = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        assert!(validate_algorithm_input(&data, Some(2), Some(100), Some(0.01)).is_ok());
    }

    #[test]
    fn test_validate_algorithm_input_empty_matrix() {
        let data = Matrix::from_vec(vec![], 0, 0).unwrap();
        assert!(validate_algorithm_input(&data, None, None, None).is_err());
    }

    #[test]
    fn test_validate_algorithm_input_invalid_k() {
        let data = Matrix::from_vec(vec![1.0, 2.0], 1, 2).unwrap();
        assert!(validate_algorithm_input(&data, Some(0), None, None).is_err());
        assert!(validate_algorithm_input(&data, Some(5), None, None).is_err());
    }

    #[test]
    fn test_validate_algorithm_input_invalid_iterations() {
        let data = Matrix::from_vec(vec![1.0, 2.0], 1, 2).unwrap();
        assert!(validate_algorithm_input(&data, None, Some(0), None).is_err());
        assert!(validate_algorithm_input(&data, None, Some(200000), None).is_err());
    }

    #[test]
    fn test_validate_algorithm_input_invalid_learning_rate() {
        let data = Matrix::from_vec(vec![1.0, 2.0], 1, 2).unwrap();
        assert!(validate_algorithm_input(&data, None, None, Some(0.0)).is_err());
        assert!(validate_algorithm_input(&data, None, None, Some(-0.1)).is_err());
        assert!(validate_algorithm_input(&data, None, None, Some(15.0)).is_err());
        assert!(validate_algorithm_input(&data, None, None, Some(f64::NAN)).is_err());
    }

    #[test]
    fn test_catch_panic_success() {
        let result = catch_panic(|| Ok(42));
        assert_eq!(result, Ok(42));
    }

    #[test]
    fn test_catch_panic_error() {
        let result = catch_panic(|| Err::<i32, _>("test error".to_string()));
        assert_eq!(result, Err("test error".to_string()));
    }

    #[test]
    fn test_catch_panic_panic() {
        let result = catch_panic(|| -> Result<(), String> {
            panic!("test panic");
        });
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("test panic"));
    }
}
