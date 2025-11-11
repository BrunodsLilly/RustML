use dioxus::prelude::*;

/// ML Algorithm type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlgorithmType {
    KMeans,
    PCA,
    LogisticRegression,
    DecisionTree,
    NaiveBayes,
    LinearRegression,
    StandardScaler,
    MinMaxScaler,
}

impl AlgorithmType {
    pub fn name(&self) -> &'static str {
        match self {
            AlgorithmType::KMeans => "K-Means Clustering",
            AlgorithmType::PCA => "Principal Component Analysis",
            AlgorithmType::LogisticRegression => "Logistic Regression",
            AlgorithmType::DecisionTree => "Decision Tree",
            AlgorithmType::NaiveBayes => "Naive Bayes",
            AlgorithmType::LinearRegression => "Linear Regression",
            AlgorithmType::StandardScaler => "Standard Scaler",
            AlgorithmType::MinMaxScaler => "Min-Max Scaler",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            AlgorithmType::KMeans => {
                "Unsupervised clustering algorithm that partitions data into k groups"
            }
            AlgorithmType::PCA => {
                "Dimensionality reduction technique using orthogonal transformation"
            }
            AlgorithmType::LogisticRegression => "Binary classification using logistic function",
            AlgorithmType::DecisionTree => {
                "Tree-based classifier using recursive splitting (CART algorithm)"
            }
            AlgorithmType::NaiveBayes => {
                "Probabilistic classifier based on Bayes' theorem with independence assumptions"
            }
            AlgorithmType::LinearRegression => "Linear model for regression problems",
            AlgorithmType::StandardScaler => {
                "Standardize features by removing mean and scaling to unit variance"
            }
            AlgorithmType::MinMaxScaler => "Scale features to a given range (default 0-1)",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            AlgorithmType::KMeans => "🎯",
            AlgorithmType::PCA => "📊",
            AlgorithmType::LogisticRegression => "📈",
            AlgorithmType::DecisionTree => "🌳",
            AlgorithmType::NaiveBayes => "🎲",
            AlgorithmType::LinearRegression => "📉",
            AlgorithmType::StandardScaler => "⚖️",
            AlgorithmType::MinMaxScaler => "📏",
        }
    }

    pub fn category(&self) -> AlgorithmCategory {
        match self {
            AlgorithmType::KMeans => AlgorithmCategory::Clustering,
            AlgorithmType::PCA => AlgorithmCategory::DimensionalityReduction,
            AlgorithmType::LogisticRegression
            | AlgorithmType::DecisionTree
            | AlgorithmType::NaiveBayes => AlgorithmCategory::Classification,
            AlgorithmType::LinearRegression => AlgorithmCategory::Regression,
            AlgorithmType::StandardScaler | AlgorithmType::MinMaxScaler => {
                AlgorithmCategory::Preprocessing
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlgorithmCategory {
    Clustering,
    Classification,
    Regression,
    DimensionalityReduction,
    Preprocessing,
}

impl AlgorithmCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            AlgorithmCategory::Clustering => "Clustering",
            AlgorithmCategory::Classification => "Classification",
            AlgorithmCategory::Regression => "Regression",
            AlgorithmCategory::DimensionalityReduction => "Dimensionality Reduction",
            AlgorithmCategory::Preprocessing => "Preprocessing",
        }
    }

    pub fn color(&self) -> &'static str {
        match self {
            AlgorithmCategory::Clustering => "#6c5ce7",
            AlgorithmCategory::Classification => "#00b894",
            AlgorithmCategory::Regression => "#0984e3",
            AlgorithmCategory::DimensionalityReduction => "#fd79a8",
            AlgorithmCategory::Preprocessing => "#fdcb6e",
        }
    }
}

/// Algorithm parameter configuration
#[derive(Clone, Debug, PartialEq)]
pub struct AlgorithmParameter {
    pub name: String,
    pub display_name: String,
    pub description: String,
    pub value_type: ParameterType,
    pub default_value: ParameterValue,
    pub current_value: ParameterValue,
    pub constraints: ParameterConstraints,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ParameterType {
    Integer,
    Float,
    Boolean,
    Choice(Vec<String>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum ParameterValue {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Choice(String),
}

impl ParameterValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ParameterValue::Float(v) => Some(*v),
            ParameterValue::Integer(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ParameterValue::Integer(v) => Some(*v),
            ParameterValue::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParameterValue::Boolean(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> String {
        match self {
            ParameterValue::Integer(v) => v.to_string(),
            ParameterValue::Float(v) => format!("{:.4}", v),
            ParameterValue::Boolean(v) => v.to_string(),
            ParameterValue::Choice(v) => v.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParameterConstraints {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    pub warning_threshold: Option<(f64, String)>,
}

impl ParameterConstraints {
    pub fn validate(&self, value: f64) -> ValidationResult {
        if let Some(min) = self.min {
            if value < min {
                return ValidationResult::Error(format!("Must be at least {}", min));
            }
        }

        if let Some(max) = self.max {
            if value > max {
                return ValidationResult::Error(format!("Must be at most {}", max));
            }
        }

        if let Some((threshold, message)) = &self.warning_threshold {
            if value > *threshold {
                return ValidationResult::Warning(message.clone());
            }
        }

        ValidationResult::Valid
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ValidationResult {
    Valid,
    Warning(String),
    Error(String),
}

/// Get default parameters for an algorithm
pub fn get_algorithm_parameters(algo: AlgorithmType) -> Vec<AlgorithmParameter> {
    match algo {
        AlgorithmType::KMeans => vec![
            AlgorithmParameter {
                name: "n_clusters".to_string(),
                display_name: "Number of Clusters (k)".to_string(),
                description: "Number of clusters to form".to_string(),
                value_type: ParameterType::Integer,
                default_value: ParameterValue::Integer(3),
                current_value: ParameterValue::Integer(3),
                constraints: ParameterConstraints {
                    min: Some(2.0),
                    max: Some(20.0),
                    step: Some(1.0),
                    warning_threshold: Some((10.0, "High k may lead to overfitting".to_string())),
                },
            },
            AlgorithmParameter {
                name: "max_iterations".to_string(),
                display_name: "Max Iterations".to_string(),
                description: "Maximum number of iterations for convergence".to_string(),
                value_type: ParameterType::Integer,
                default_value: ParameterValue::Integer(100),
                current_value: ParameterValue::Integer(100),
                constraints: ParameterConstraints {
                    min: Some(1.0),
                    max: Some(1000.0),
                    step: Some(10.0),
                    warning_threshold: Some((
                        500.0,
                        "High iteration count may be slow".to_string(),
                    )),
                },
            },
            AlgorithmParameter {
                name: "tolerance".to_string(),
                display_name: "Convergence Tolerance".to_string(),
                description: "Stop when centroid movement is below this threshold".to_string(),
                value_type: ParameterType::Float,
                default_value: ParameterValue::Float(1e-4),
                current_value: ParameterValue::Float(1e-4),
                constraints: ParameterConstraints {
                    min: Some(1e-10),
                    max: Some(1e-1),
                    step: Some(1e-5),
                    warning_threshold: None,
                },
            },
        ],
        AlgorithmType::PCA => vec![AlgorithmParameter {
            name: "n_components".to_string(),
            display_name: "Number of Components".to_string(),
            description: "Number of principal components to keep".to_string(),
            value_type: ParameterType::Integer,
            default_value: ParameterValue::Integer(2),
            current_value: ParameterValue::Integer(2),
            constraints: ParameterConstraints {
                min: Some(1.0),
                max: Some(50.0),
                step: Some(1.0),
                warning_threshold: None,
            },
        }],
        AlgorithmType::LogisticRegression => vec![
            AlgorithmParameter {
                name: "learning_rate".to_string(),
                display_name: "Learning Rate".to_string(),
                description: "Step size for gradient descent".to_string(),
                value_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.01),
                current_value: ParameterValue::Float(0.01),
                constraints: ParameterConstraints {
                    min: Some(1e-5),
                    max: Some(1.0),
                    step: Some(0.001),
                    warning_threshold: Some((
                        0.1,
                        "High learning rate may cause instability".to_string(),
                    )),
                },
            },
            AlgorithmParameter {
                name: "max_iterations".to_string(),
                display_name: "Max Iterations".to_string(),
                description: "Maximum number of gradient descent iterations".to_string(),
                value_type: ParameterType::Integer,
                default_value: ParameterValue::Integer(1000),
                current_value: ParameterValue::Integer(1000),
                constraints: ParameterConstraints {
                    min: Some(10.0),
                    max: Some(10000.0),
                    step: Some(100.0),
                    warning_threshold: Some((5000.0, "Many iterations may be slow".to_string())),
                },
            },
        ],
        AlgorithmType::DecisionTree => vec![
            AlgorithmParameter {
                name: "max_depth".to_string(),
                display_name: "Max Depth".to_string(),
                description: "Maximum depth of the decision tree".to_string(),
                value_type: ParameterType::Integer,
                default_value: ParameterValue::Integer(10),
                current_value: ParameterValue::Integer(10),
                constraints: ParameterConstraints {
                    min: Some(1.0),
                    max: Some(50.0),
                    step: Some(1.0),
                    warning_threshold: Some((30.0, "Very deep trees may overfit".to_string())),
                },
            },
            AlgorithmParameter {
                name: "min_samples_split".to_string(),
                display_name: "Min Samples Split".to_string(),
                description: "Minimum samples required to split a node".to_string(),
                value_type: ParameterType::Integer,
                default_value: ParameterValue::Integer(2),
                current_value: ParameterValue::Integer(2),
                constraints: ParameterConstraints {
                    min: Some(2.0),
                    max: Some(100.0),
                    step: Some(1.0),
                    warning_threshold: None,
                },
            },
            AlgorithmParameter {
                name: "min_samples_leaf".to_string(),
                display_name: "Min Samples Leaf".to_string(),
                description: "Minimum samples required in a leaf node".to_string(),
                value_type: ParameterType::Integer,
                default_value: ParameterValue::Integer(1),
                current_value: ParameterValue::Integer(1),
                constraints: ParameterConstraints {
                    min: Some(1.0),
                    max: Some(50.0),
                    step: Some(1.0),
                    warning_threshold: None,
                },
            },
        ],
        AlgorithmType::NaiveBayes => vec![
            // Naive Bayes has no configurable hyperparameters in our implementation
            // (epsilon is hardcoded for numerical stability)
        ],
        AlgorithmType::LinearRegression => vec![
            AlgorithmParameter {
                name: "learning_rate".to_string(),
                display_name: "Learning Rate".to_string(),
                description: "Step size for gradient descent".to_string(),
                value_type: ParameterType::Float,
                default_value: ParameterValue::Float(0.01),
                current_value: ParameterValue::Float(0.01),
                constraints: ParameterConstraints {
                    min: Some(1e-5),
                    max: Some(1.0),
                    step: Some(0.001),
                    warning_threshold: Some((0.1, "High learning rate may diverge".to_string())),
                },
            },
            AlgorithmParameter {
                name: "max_iterations".to_string(),
                display_name: "Max Iterations".to_string(),
                description: "Maximum number of gradient descent iterations".to_string(),
                value_type: ParameterType::Integer,
                default_value: ParameterValue::Integer(1000),
                current_value: ParameterValue::Integer(1000),
                constraints: ParameterConstraints {
                    min: Some(10.0),
                    max: Some(10000.0),
                    step: Some(100.0),
                    warning_threshold: None,
                },
            },
        ],
        AlgorithmType::StandardScaler | AlgorithmType::MinMaxScaler => vec![],
    }
}

/// Props for AlgorithmConfigurator component
#[derive(Props, Clone, PartialEq)]
pub struct AlgorithmConfiguratorProps {
    /// Selected algorithm
    algorithm: AlgorithmType,
    /// Current parameters
    #[props(default)]
    parameters: Option<Vec<AlgorithmParameter>>,
    /// Callback when parameters change
    on_parameters_change: EventHandler<Vec<AlgorithmParameter>>,
    /// Show advanced options
    #[props(default = false)]
    show_advanced: bool,
}

/// Interactive algorithm configuration component with real-time parameter tuning
///
/// # Example
/// ```rust
/// let mut selected_algo = use_signal(|| AlgorithmType::KMeans);
/// let mut params = use_signal(|| get_algorithm_parameters(AlgorithmType::KMeans));
///
/// rsx! {
///     AlgorithmConfigurator {
///         algorithm: selected_algo(),
///         parameters: Some(params()),
///         on_parameters_change: move |new_params| {
///             params.set(new_params);
///         }
///     }
/// }
/// ```
#[component]
pub fn AlgorithmConfigurator(props: AlgorithmConfiguratorProps) -> Element {
    let params = use_signal(|| {
        props
            .parameters
            .clone()
            .unwrap_or_else(|| get_algorithm_parameters(props.algorithm))
    });

    rsx! {
        div { class: "algorithm-configurator",
            // Algorithm header
            div { class: "algo-header",
                span { class: "algo-icon", "{props.algorithm.icon()}" }
                div { class: "algo-info",
                    h3 { "{props.algorithm.name()}" }
                    p { class: "algo-description", "{props.algorithm.description()}" }
                    span {
                        class: "algo-category",
                        style: "background-color: {props.algorithm.category().color()}",
                        "{props.algorithm.category().as_str()}"
                    }
                }
            }

            // Parameters section
            {
                let params_vec = params.read().clone();
                if !params_vec.is_empty() {
                    rsx! {
                        div { class: "parameters-section",
                            h4 { "⚙️ Configuration" }

                            for param in params_vec.iter() {
                                ParameterControl {
                                    parameter: param.clone(),
                                    on_change: {
                                        let params_clone = params_vec.clone();
                                        let handler = props.on_parameters_change.clone();
                                        let param_name = param.name.clone();
                                        move |new_value| {
                                            let mut updated_params = params_clone.clone();
                                            if let Some(p) = updated_params.iter_mut().find(|p| p.name == param_name) {
                                                p.current_value = new_value;
                                            }
                                            handler.call(updated_params);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    rsx! {
                        div { class: "no-parameters",
                            "ℹ️ This algorithm has no configurable parameters"
                        }
                    }
                }
            }

            // Quick presets
            div { class: "presets-section",
                h4 { "🎛️ Presets" }
                div { class: "preset-buttons",
                    button {
                        class: "preset-btn conservative",
                        onclick: move |_| {
                            let conservative_params = get_conservative_preset(props.algorithm);
                            props.on_parameters_change.call(conservative_params);
                        },
                        "🐢 Conservative"
                    }
                    button {
                        class: "preset-btn balanced",
                        onclick: move |_| {
                            let default_params = get_algorithm_parameters(props.algorithm);
                            props.on_parameters_change.call(default_params);
                        },
                        "⚖️ Balanced"
                    }
                    button {
                        class: "preset-btn aggressive",
                        onclick: move |_| {
                            let aggressive_params = get_aggressive_preset(props.algorithm);
                            props.on_parameters_change.call(aggressive_params);
                        },
                        "🚀 Aggressive"
                    }
                }
            }
        }
    }
}

fn get_conservative_preset(algo: AlgorithmType) -> Vec<AlgorithmParameter> {
    let mut params = get_algorithm_parameters(algo);
    for param in &mut params {
        match param.name.as_str() {
            "learning_rate" => param.current_value = ParameterValue::Float(0.001),
            "max_iterations" => param.current_value = ParameterValue::Integer(500),
            "n_clusters" => param.current_value = ParameterValue::Integer(2),
            _ => {}
        }
    }
    params
}

fn get_aggressive_preset(algo: AlgorithmType) -> Vec<AlgorithmParameter> {
    let mut params = get_algorithm_parameters(algo);
    for param in &mut params {
        match param.name.as_str() {
            "learning_rate" => param.current_value = ParameterValue::Float(0.1),
            "max_iterations" => param.current_value = ParameterValue::Integer(2000),
            "n_clusters" => param.current_value = ParameterValue::Integer(8),
            _ => {}
        }
    }
    params
}

/// Props for ParameterControl component
#[derive(Props, Clone, PartialEq)]
pub struct ParameterControlProps {
    parameter: AlgorithmParameter,
    on_change: EventHandler<ParameterValue>,
}

/// Individual parameter control with validation
#[component]
pub fn ParameterControl(props: ParameterControlProps) -> Element {
    let validation = if let Some(val) = props.parameter.current_value.as_f64() {
        props.parameter.constraints.validate(val)
    } else {
        ValidationResult::Valid
    };

    // Extract options if it's a Choice type
    let options = match &props.parameter.value_type {
        ParameterType::Choice(opts) => Some(opts.clone()),
        _ => None,
    };

    rsx! {
        div { class: "parameter-control",
            div { class: "param-header",
                label { "{props.parameter.display_name}" }
                span { class: "param-value", "{props.parameter.current_value.as_str()}" }
            }

            match &props.parameter.value_type {
                ParameterType::Integer | ParameterType::Float => {
                    rsx! {
                        input {
                            r#type: "range",
                            class: "param-slider",
                            min: "{props.parameter.constraints.min.unwrap_or(0.0)}",
                            max: "{props.parameter.constraints.max.unwrap_or(100.0)}",
                            step: "{props.parameter.constraints.step.unwrap_or(1.0)}",
                            value: "{props.parameter.current_value.as_f64().unwrap_or(0.0)}",
                            oninput: move |evt| {
                                let val_str = evt.value();
                                if let Ok(val) = val_str.parse::<f64>() {
                                    let new_value = match props.parameter.value_type {
                                        ParameterType::Integer => ParameterValue::Integer(val as i64),
                                        ParameterType::Float => ParameterValue::Float(val),
                                        _ => return,
                                    };
                                    props.on_change.call(new_value);
                                }
                            }
                        }
                    }
                },
                ParameterType::Boolean => {
                    rsx! {
                        input {
                            r#type: "checkbox",
                            class: "param-checkbox",
                            checked: props.parameter.current_value.as_bool().unwrap_or(false),
                            onchange: move |evt| {
                                props.on_change.call(ParameterValue::Boolean(evt.checked()));
                            }
                        }
                    }
                },
                ParameterType::Choice(_) => {
                    let opts = options.clone().unwrap_or_default();
                    rsx! {
                        select {
                            class: "param-select",
                            value: "{props.parameter.current_value.as_str()}",
                            onchange: move |evt| {
                                props.on_change.call(ParameterValue::Choice(evt.value()));
                            },
                            for opt in opts.iter() {
                                option { value: "{opt}", "{opt}" }
                            }
                        }
                    }
                }
            }

            p { class: "param-description", "{props.parameter.description}" }

            // Validation feedback
            match validation {
                ValidationResult::Warning(msg) => rsx! {
                    div { class: "param-validation warning",
                        "⚠️ {msg}"
                    }
                },
                ValidationResult::Error(msg) => rsx! {
                    div { class: "param-validation error",
                        "❌ {msg}"
                    }
                },
                ValidationResult::Valid => rsx! {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_names() {
        assert_eq!(AlgorithmType::KMeans.name(), "K-Means Clustering");
        assert_eq!(AlgorithmType::PCA.name(), "Principal Component Analysis");
    }

    #[test]
    fn test_parameter_value_conversions() {
        let int_val = ParameterValue::Integer(42);
        assert_eq!(int_val.as_i64(), Some(42));
        assert_eq!(int_val.as_f64(), Some(42.0));

        let float_val = ParameterValue::Float(3.14);
        assert_eq!(float_val.as_f64(), Some(3.14));
    }

    #[test]
    fn test_parameter_validation() {
        let constraints = ParameterConstraints {
            min: Some(0.0),
            max: Some(10.0),
            step: None,
            warning_threshold: Some((8.0, "High value".to_string())),
        };

        assert!(matches!(constraints.validate(5.0), ValidationResult::Valid));
        assert!(matches!(
            constraints.validate(9.0),
            ValidationResult::Warning(_)
        ));
        assert!(matches!(
            constraints.validate(11.0),
            ValidationResult::Error(_)
        ));
        assert!(matches!(
            constraints.validate(-1.0),
            ValidationResult::Error(_)
        ));
    }

    #[test]
    fn test_get_kmeans_parameters() {
        let params = get_algorithm_parameters(AlgorithmType::KMeans);
        assert_eq!(params.len(), 3);
        assert_eq!(params[0].name, "n_clusters");
        assert_eq!(params[1].name, "max_iterations");
        assert_eq!(params[2].name, "tolerance");
    }

    #[test]
    fn test_conservative_preset() {
        let params = get_conservative_preset(AlgorithmType::LogisticRegression);
        let lr_param = params.iter().find(|p| p.name == "learning_rate").unwrap();
        assert_eq!(lr_param.current_value.as_f64(), Some(0.001));
    }
}
