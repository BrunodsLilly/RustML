use dioxus::prelude::*;

/// Validation state for form inputs
#[derive(Clone, PartialEq, Debug)]
pub enum ValidationState {
    /// No validation performed yet
    NotValidated,
    /// Input is valid
    Valid,
    /// Input has errors
    Invalid { message: String },
    /// Warning (input is valid but not recommended)
    Warning { message: String },
}

impl ValidationState {
    /// Check if the state is valid (either Valid or Warning)
    pub fn is_valid(&self) -> bool {
        matches!(
            self,
            ValidationState::Valid | ValidationState::Warning { .. }
        )
    }

    /// Get error message if invalid
    pub fn error_message(&self) -> Option<&str> {
        match self {
            ValidationState::Invalid { message } => Some(message),
            _ => None,
        }
    }

    /// Get warning message if warning
    pub fn warning_message(&self) -> Option<&str> {
        match self {
            ValidationState::Warning { message } => Some(message),
            _ => None,
        }
    }
}

/// Props for ValidatedInput component
#[derive(Props, Clone, PartialEq)]
pub struct ValidatedInputProps {
    /// Label for the input
    label: String,
    /// Input type (text, number, range, etc.)
    #[props(default = "text")]
    input_type: &'static str,
    /// Current value
    value: String,
    /// Callback when value changes
    on_change: EventHandler<String>,
    /// Optional validation state (external control)
    #[props(default)]
    validation_state: Option<ValidationState>,
    /// Help text shown below input
    #[props(default = "")]
    help_text: &'static str,
    /// Minimum value (for number/range inputs)
    #[props(default)]
    min: Option<String>,
    /// Maximum value (for number/range inputs)
    #[props(default)]
    max: Option<String>,
    /// Step value (for number/range inputs)
    #[props(default)]
    step: Option<String>,
    /// Placeholder text
    #[props(default = "")]
    placeholder: &'static str,
    /// Whether the input is disabled
    #[props(default = false)]
    disabled: bool,
    /// Whether the input is required
    #[props(default = false)]
    required: bool,
}

/// Validated input component with real-time feedback
///
/// # Example
/// ```rust
/// let mut value = use_signal(|| "0".to_string());
/// let mut validation = use_signal(|| ValidationState::NotValidated);
///
/// rsx! {
///     ValidatedInput {
///         label: "Learning Rate",
///         input_type: "number",
///         value: "{value}",
///         min: "0.0001",
///         max: "1.0",
///         step: "0.0001",
///         help_text: "Controls how quickly the model learns",
///         required: true,
///         on_change: move |new_val| {
///             value.set(new_val.clone());
///             // Validate
///             let state = if let Ok(lr) = new_val.parse::<f64>() {
///                 if lr <= 0.0 {
///                     ValidationState::Invalid {
///                         message: "Learning rate must be positive".to_string()
///                     }
///                 } else if lr > 1.0 {
///                     ValidationState::Warning {
///                         message: "Learning rate above 1.0 may cause instability".to_string()
///                     }
///                 } else {
///                     ValidationState::Valid
///                 }
///             } else {
///                 ValidationState::Invalid {
///                     message: "Must be a valid number".to_string()
///                 }
///             };
///             validation.set(state);
///         },
///     }
/// }
/// ```
#[component]
pub fn ValidatedInput(props: ValidatedInputProps) -> Element {
    let handle_change = move |evt: Event<FormData>| {
        let new_value = evt.value();
        props.on_change.call(new_value);
    };

    // Use external validation state if provided, otherwise default to NotValidated
    let current_state = props
        .validation_state
        .as_ref()
        .unwrap_or(&ValidationState::NotValidated);

    rsx! {
        div { class: "input-group validated",
            label { class: "input-label",
                "{props.label}"
                if props.required {
                    span { class: "required-indicator", " *" }
                }
            }

            {
                let input_class = match current_state {
                    ValidationState::NotValidated => "validated-input",
                    ValidationState::Valid => "validated-input valid",
                    ValidationState::Invalid { .. } => "validated-input invalid",
                    ValidationState::Warning { .. } => "validated-input warning",
                };

                rsx! {
                    input {
                        class: input_class,
                        r#type: props.input_type,
                        value: "{props.value}",
                        placeholder: props.placeholder,
                        disabled: props.disabled,
                        required: props.required,
                        min: props.min.as_deref().unwrap_or(""),
                        max: props.max.as_deref().unwrap_or(""),
                        step: props.step.as_deref().unwrap_or(""),
                        oninput: handle_change,
                    }
                }
            }

            // Validation feedback
            {
                match current_state {
                    ValidationState::Invalid { message } => rsx! {
                        div { class: "validation-message error",
                            span { class: "validation-icon", "❌ " }
                            "{message}"
                        }
                    },
                    ValidationState::Warning { message } => rsx! {
                        div { class: "validation-message warning",
                            span { class: "validation-icon", "⚠️ " }
                            "{message}"
                        }
                    },
                    ValidationState::Valid => rsx! {
                        div { class: "validation-message success",
                            span { class: "validation-icon", "✅ " }
                            "Valid"
                        }
                    },
                    ValidationState::NotValidated => rsx! { }
                }
            }

            // Help text
            if !props.help_text.is_empty() {
                p { class: "help-text", "{props.help_text}" }
            }
        }
    }
}

/// Common validation functions
pub mod validators {
    use super::ValidationState;

    /// Validate that a string is a valid positive number
    pub fn positive_number(value: &str) -> ValidationState {
        match value.parse::<f64>() {
            Ok(num) if num > 0.0 && num.is_finite() => ValidationState::Valid,
            Ok(num) if num <= 0.0 => ValidationState::Invalid {
                message: "Must be a positive number".to_string(),
            },
            Ok(_) => ValidationState::Invalid {
                message: "Must be a finite number".to_string(),
            },
            Err(_) => ValidationState::Invalid {
                message: "Must be a valid number".to_string(),
            },
        }
    }

    /// Validate learning rate with warnings for extreme values
    pub fn learning_rate(value: &str) -> ValidationState {
        match value.parse::<f64>() {
            Ok(lr) if lr <= 0.0 => ValidationState::Invalid {
                message: "Learning rate must be positive".to_string(),
            },
            Ok(lr) if lr > 10.0 => ValidationState::Warning {
                message: "Very high learning rate may cause instability".to_string(),
            },
            Ok(lr) if lr < 0.0001 => ValidationState::Warning {
                message: "Very low learning rate may converge slowly".to_string(),
            },
            Ok(_) => ValidationState::Valid,
            Err(_) => ValidationState::Invalid {
                message: "Must be a valid number".to_string(),
            },
        }
    }

    /// Validate integer in range
    pub fn integer_in_range(min: usize, max: usize) -> impl Fn(&str) -> ValidationState {
        move |value: &str| match value.parse::<usize>() {
            Ok(num) if num < min => ValidationState::Invalid {
                message: format!("Must be at least {}", min),
            },
            Ok(num) if num > max => ValidationState::Invalid {
                message: format!("Must be at most {}", max),
            },
            Ok(_) => ValidationState::Valid,
            Err(_) => ValidationState::Invalid {
                message: "Must be a valid integer".to_string(),
            },
        }
    }

    /// Validate k_clusters with recommendations
    pub fn k_clusters(value: &str, n_samples: usize) -> ValidationState {
        match value.parse::<usize>() {
            Ok(k) if k == 0 => ValidationState::Invalid {
                message: "Number of clusters must be at least 1".to_string(),
            },
            Ok(k) if k > n_samples => ValidationState::Invalid {
                message: format!("Cannot exceed number of samples ({})", n_samples),
            },
            Ok(k) if k > n_samples / 2 => ValidationState::Warning {
                message: "High k may lead to overfitting".to_string(),
            },
            Ok(k) if k == 1 => ValidationState::Warning {
                message: "Single cluster is rarely useful".to_string(),
            },
            Ok(_) => ValidationState::Valid,
            Err(_) => ValidationState::Invalid {
                message: "Must be a valid integer".to_string(),
            },
        }
    }

    /// Validate max iterations
    pub fn max_iterations(value: &str) -> ValidationState {
        match value.parse::<usize>() {
            Ok(iter) if iter == 0 => ValidationState::Invalid {
                message: "Must be at least 1 iteration".to_string(),
            },
            Ok(iter) if iter > 100000 => ValidationState::Warning {
                message: "Very high iteration count may be slow".to_string(),
            },
            Ok(_) => ValidationState::Valid,
            Err(_) => ValidationState::Invalid {
                message: "Must be a valid integer".to_string(),
            },
        }
    }

    /// Validate CSV file size (in bytes)
    pub fn csv_file_size(size: usize) -> ValidationState {
        const MAX_SIZE: usize = 5 * 1024 * 1024; // 5MB
        const WARN_SIZE: usize = 2 * 1024 * 1024; // 2MB

        if size > MAX_SIZE {
            ValidationState::Invalid {
                message: format!(
                    "File too large (max 5MB, got {:.1}MB)",
                    size as f64 / 1024.0 / 1024.0
                ),
            }
        } else if size > WARN_SIZE {
            ValidationState::Warning {
                message: format!(
                    "Large file ({:.1}MB) may be slow to process",
                    size as f64 / 1024.0 / 1024.0
                ),
            }
        } else {
            ValidationState::Valid
        }
    }

    /// Validate CSV dimensions (rows × columns)
    pub fn csv_dimensions(rows: usize, cols: usize) -> ValidationState {
        const MAX_ROWS: usize = 10000;
        const MAX_COLS: usize = 100;
        const WARN_ROWS: usize = 5000;
        const WARN_COLS: usize = 50;

        if rows > MAX_ROWS {
            ValidationState::Invalid {
                message: format!("Too many rows (max 10,000, got {})", rows),
            }
        } else if cols > MAX_COLS {
            ValidationState::Invalid {
                message: format!("Too many columns (max 100, got {})", cols),
            }
        } else if rows > WARN_ROWS || cols > WARN_COLS {
            ValidationState::Warning {
                message: format!("Large dataset ({}×{}) may be slow", rows, cols),
            }
        } else if rows < 10 {
            ValidationState::Warning {
                message: "Very small dataset may not produce reliable results".to_string(),
            }
        } else {
            ValidationState::Valid
        }
    }
}

#[cfg(test)]
mod tests {
    use super::validators::*;
    use super::ValidationState;

    #[test]
    fn test_positive_number_valid() {
        assert!(matches!(positive_number("1.5"), ValidationState::Valid));
        assert!(matches!(positive_number("0.001"), ValidationState::Valid));
    }

    #[test]
    fn test_positive_number_invalid() {
        assert!(matches!(
            positive_number("-1.0"),
            ValidationState::Invalid { .. }
        ));
        assert!(matches!(
            positive_number("0"),
            ValidationState::Invalid { .. }
        ));
        assert!(matches!(
            positive_number("abc"),
            ValidationState::Invalid { .. }
        ));
    }

    #[test]
    fn test_learning_rate_valid() {
        assert!(matches!(learning_rate("0.01"), ValidationState::Valid));
        assert!(matches!(learning_rate("1.0"), ValidationState::Valid));
    }

    #[test]
    fn test_learning_rate_warning() {
        assert!(matches!(
            learning_rate("15.0"),
            ValidationState::Warning { .. }
        ));
        assert!(matches!(
            learning_rate("0.00001"),
            ValidationState::Warning { .. }
        ));
    }

    #[test]
    fn test_integer_in_range() {
        let validator = integer_in_range(1, 10);
        assert!(matches!(validator("5"), ValidationState::Valid));
        assert!(matches!(validator("0"), ValidationState::Invalid { .. }));
        assert!(matches!(validator("11"), ValidationState::Invalid { .. }));
    }

    #[test]
    fn test_k_clusters_valid() {
        assert!(matches!(k_clusters("3", 100), ValidationState::Valid));
        assert!(matches!(k_clusters("10", 100), ValidationState::Valid));
    }

    #[test]
    fn test_k_clusters_invalid() {
        assert!(matches!(
            k_clusters("0", 100),
            ValidationState::Invalid { .. }
        ));
        assert!(matches!(
            k_clusters("101", 100),
            ValidationState::Invalid { .. }
        ));
    }

    #[test]
    fn test_k_clusters_warning() {
        assert!(matches!(
            k_clusters("1", 100),
            ValidationState::Warning { .. }
        ));
        assert!(matches!(
            k_clusters("60", 100),
            ValidationState::Warning { .. }
        ));
    }

    #[test]
    fn test_csv_file_size() {
        assert!(matches!(csv_file_size(1024), ValidationState::Valid));
        assert!(matches!(
            csv_file_size(3 * 1024 * 1024),
            ValidationState::Warning { .. }
        ));
        assert!(matches!(
            csv_file_size(6 * 1024 * 1024),
            ValidationState::Invalid { .. }
        ));
    }

    #[test]
    fn test_csv_dimensions() {
        assert!(matches!(csv_dimensions(100, 10), ValidationState::Valid));
        assert!(matches!(
            csv_dimensions(5, 5),
            ValidationState::Warning { .. }
        ));
        assert!(matches!(
            csv_dimensions(6000, 10),
            ValidationState::Warning { .. }
        ));
        assert!(matches!(
            csv_dimensions(11000, 10),
            ValidationState::Invalid { .. }
        ));
        assert!(matches!(
            csv_dimensions(100, 101),
            ValidationState::Invalid { .. }
        ));
    }
}
