use dioxus::prelude::*;

/// Feature metadata for selection
#[derive(Clone, Debug, PartialEq)]
pub struct Feature {
    pub name: String,
    pub dtype: DataType,
    pub unique_values: usize,
    pub missing_count: usize,
    pub total_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DataType {
    Numeric,
    Categorical,
    Binary,
    Text,
    DateTime,
    Unknown,
}

impl DataType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DataType::Numeric => "Numeric",
            DataType::Categorical => "Categorical",
            DataType::Binary => "Binary",
            DataType::Text => "Text",
            DataType::DateTime => "DateTime",
            DataType::Unknown => "Unknown",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            DataType::Numeric => "📊",
            DataType::Categorical => "🏷️",
            DataType::Binary => "🔀",
            DataType::Text => "📝",
            DataType::DateTime => "📅",
            DataType::Unknown => "❓",
        }
    }

    pub fn color(&self) -> &'static str {
        match self {
            DataType::Numeric => "#6c5ce7",
            DataType::Categorical => "#00b894",
            DataType::Binary => "#fdcb6e",
            DataType::Text => "#fd79a8",
            DataType::DateTime => "#74b9ff",
            DataType::Unknown => "#b2bec3",
        }
    }
}

impl Feature {
    /// Infer data type from column values
    pub fn infer_dtype(values: &[String]) -> DataType {
        if values.is_empty() {
            return DataType::Unknown;
        }

        let non_empty: Vec<&String> = values.iter().filter(|s| !s.trim().is_empty()).collect();

        if non_empty.is_empty() {
            return DataType::Unknown;
        }

        // Check if all values are numeric
        let numeric_count = non_empty
            .iter()
            .filter(|s| s.parse::<f64>().is_ok())
            .count();

        if numeric_count == non_empty.len() {
            // Check if binary (only 0 and 1)
            let unique_values: std::collections::HashSet<String> =
                non_empty.iter().map(|s| (*s).clone()).collect();

            if unique_values.len() == 2
                && unique_values.contains("0")
                && unique_values.contains("1")
            {
                return DataType::Binary;
            }
            return DataType::Numeric;
        }

        // Check if categorical (low cardinality)
        let unique_values: std::collections::HashSet<String> =
            non_empty.iter().map(|s| (*s).clone()).collect();

        let cardinality = unique_values.len();
        let cardinality_ratio = cardinality as f64 / non_empty.len() as f64;

        // If cardinality < 10 or ratio < 0.05, it's categorical
        if cardinality < 10 || cardinality_ratio < 0.05 {
            return DataType::Categorical;
        }

        // Check for date/time patterns (simple heuristic)
        let datetime_patterns = ["-", "/", ":", "T"];
        let datetime_count = non_empty
            .iter()
            .filter(|s| datetime_patterns.iter().any(|p| s.contains(p)))
            .count();

        if datetime_count as f64 / non_empty.len() as f64 > 0.8 {
            return DataType::DateTime;
        }

        // Default to text
        DataType::Text
    }

    pub fn missing_percent(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            (self.missing_count as f64 / self.total_count as f64) * 100.0
        }
    }
}

/// Props for FeatureSelector component
#[derive(Props, Clone, PartialEq)]
pub struct FeatureSelectorProps {
    /// Available features
    features: Vec<Feature>,
    /// Currently selected feature names
    selected: Vec<String>,
    /// Callback when selection changes
    on_selection_change: EventHandler<Vec<String>>,
    /// Target column (cannot be selected as feature)
    #[props(default)]
    target_column: Option<String>,
    /// Enable drag-and-drop reordering
    #[props(default = true)]
    enable_drag_drop: bool,
    /// Show feature statistics
    #[props(default = true)]
    show_stats: bool,
}

/// Feature selector component with drag-drop for ML model feature selection
///
/// # Example
/// ```rust
/// let mut selected = use_signal(|| vec!["age".to_string(), "income".to_string()]);
///
/// rsx! {
///     FeatureSelector {
///         features: all_features,
///         selected: selected(),
///         target_column: Some("price".to_string()),
///         on_selection_change: move |new_selection| {
///             selected.set(new_selection);
///         }
///     }
/// }
/// ```
#[component]
pub fn FeatureSelector(props: FeatureSelectorProps) -> Element {
    let mut drag_source = use_signal(|| None::<usize>);
    let mut search_query = use_signal(|| String::new());

    // Filter features based on search
    let filtered_features: Vec<&Feature> = props
        .features
        .iter()
        .filter(|f| {
            let query = search_query.read().to_lowercase();
            query.is_empty() || f.name.to_lowercase().contains(&query)
        })
        .collect();

    // Split into available and selected
    let available: Vec<&Feature> = filtered_features
        .iter()
        .filter(|f| {
            !props.selected.contains(&f.name)
                && props
                    .target_column
                    .as_ref()
                    .map_or(true, |target| &f.name != target)
        })
        .copied()
        .collect();

    let selected_features: Vec<&Feature> = props
        .selected
        .iter()
        .filter_map(|name| props.features.iter().find(|f| &f.name == name))
        .collect();

    // Handle feature click (toggle selection) - defined inline below to avoid move issues

    rsx! {
        div { class: "feature-selector-container",
            h3 { class: "feature-selector-title", "🎯 Feature Selection" }

            // Search bar
            div { class: "feature-search",
                input {
                    r#type: "text",
                    class: "search-input",
                    placeholder: "🔍 Search features...",
                    value: "{search_query}",
                    oninput: move |evt| search_query.set(evt.value()),
                }
            }

            // Selection summary
            div { class: "selection-summary",
                div { class: "summary-card",
                    span { class: "summary-label", "Total Features" }
                    span { class: "summary-value", "{props.features.len()}" }
                }
                div { class: "summary-card selected",
                    span { class: "summary-label", "Selected" }
                    span { class: "summary-value", "{props.selected.len()}" }
                }
                div { class: "summary-card available",
                    span { class: "summary-label", "Available" }
                    span { class: "summary-value", "{available.len()}" }
                }
            }

            // Two-column layout
            div { class: "feature-columns",
                // Available features
                div { class: "feature-column",
                    h4 { "📦 Available Features ({available.len()})" }
                    div { class: "feature-list",
                        if available.is_empty() {
                            div { class: "empty-state", "All features selected or no matches" }
                        } else {
                            for feature in available.iter() {
                                {
                                    let name = feature.name.clone();
                                    let handler = props.on_selection_change.clone();
                                    let current_selected = props.selected.clone();

                                    rsx! {
                                        FeatureCard {
                                            feature: (*feature).clone(),
                                            selected: false,
                                            show_stats: props.show_stats,
                                            onclick: move |_| {
                                                let mut new_selected = current_selected.clone();
                                                if let Some(pos) = new_selected.iter().position(|n| n == &name) {
                                                    new_selected.remove(pos);
                                                } else {
                                                    new_selected.push(name.clone());
                                                }
                                                handler.call(new_selected);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Selected features (reorderable)
                div { class: "feature-column selected-column",
                    h4 { "✅ Selected Features ({selected_features.len()})" }

                    if props.enable_drag_drop {
                        p { class: "drag-hint", "💡 Drag to reorder feature importance" }
                    }

                    div { class: "feature-list",
                        if selected_features.is_empty() {
                            div { class: "empty-state", "No features selected yet" }
                        } else {
                            for (idx, feature) in selected_features.iter().enumerate() {
                                {
                                    let name = feature.name.clone();
                                    let draggable = props.enable_drag_drop;
                                    let handler = props.on_selection_change.clone();
                                    let current_selected = props.selected.clone();
                                    let handler_click = handler.clone();
                                    let current_selected_click = current_selected.clone();

                                    rsx! {
                                        div {
                                            key: "{idx}",
                                            class: "feature-card-wrapper",
                                            draggable: "{draggable}",
                                            ondragstart: move |_| {
                                                drag_source.set(Some(idx));
                                            },
                                            ondragover: move |evt| evt.prevent_default(),
                                            ondrop: move |_| {
                                                if let Some(source_idx) = *drag_source.read() {
                                                    if source_idx != idx {
                                                        let mut new_selected = current_selected.clone();
                                                        let item = new_selected.remove(source_idx);
                                                        new_selected.insert(idx, item);
                                                        handler.call(new_selected);
                                                    }
                                                }
                                                drag_source.set(None);
                                            },

                                            div { class: "feature-order", "{idx + 1}" }

                                            FeatureCard {
                                                feature: (*feature).clone(),
                                                selected: true,
                                                show_stats: props.show_stats,
                                                onclick: move |_| {
                                                    let mut new_selected = current_selected_click.clone();
                                                    if let Some(pos) = new_selected.iter().position(|n| n == &name) {
                                                        new_selected.remove(pos);
                                                    } else {
                                                        new_selected.push(name.clone());
                                                    }
                                                    handler_click.call(new_selected);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Target column indicator
            if let Some(ref target) = props.target_column {
                div { class: "target-indicator",
                    "🎯 Target: {target}"
                }
            }
        }
    }
}

/// Props for FeatureCard component
#[derive(Props, Clone, PartialEq)]
pub struct FeatureCardProps {
    feature: Feature,
    selected: bool,
    show_stats: bool,
    onclick: EventHandler<()>,
}

/// Individual feature card
#[component]
pub fn FeatureCard(props: FeatureCardProps) -> Element {
    let card_class = if props.selected {
        "feature-card selected"
    } else {
        "feature-card"
    };

    rsx! {
        div {
            class: card_class,
            onclick: move |_| props.onclick.call(()),

            // Feature header
            div { class: "feature-header",
                span { class: "feature-icon", "{props.feature.dtype.icon()}" }
                span { class: "feature-name", "{props.feature.name}" }
                span {
                    class: "feature-type",
                    style: "background-color: {props.feature.dtype.color()}",
                    "{props.feature.dtype.as_str()}"
                }
            }

            // Feature stats
            if props.show_stats {
                div { class: "feature-stats",
                    div { class: "stat-item",
                        span { class: "stat-label", "Unique" }
                        span { class: "stat-value", "{props.feature.unique_values}" }
                    }
                    div { class: "stat-item",
                        span { class: "stat-label", "Missing" }
                        span {
                            class: if props.feature.missing_count > 0 { "stat-value warning" } else { "stat-value" },
                            "{props.feature.missing_percent():.1}%"
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_numeric_dtype() {
        let values = vec!["1.5".to_string(), "2.3".to_string(), "3.7".to_string()];
        let dtype = Feature::infer_dtype(&values);
        assert_eq!(dtype, DataType::Numeric);
    }

    #[test]
    fn test_infer_binary_dtype() {
        let values = vec![
            "0".to_string(),
            "1".to_string(),
            "0".to_string(),
            "1".to_string(),
        ];
        let dtype = Feature::infer_dtype(&values);
        assert_eq!(dtype, DataType::Binary);
    }

    #[test]
    fn test_infer_categorical_dtype() {
        let values = vec![
            "red".to_string(),
            "blue".to_string(),
            "red".to_string(),
            "green".to_string(),
        ];
        let dtype = Feature::infer_dtype(&values);
        assert_eq!(dtype, DataType::Categorical);
    }

    #[test]
    fn test_missing_percent() {
        let feature = Feature {
            name: "test".to_string(),
            dtype: DataType::Numeric,
            unique_values: 10,
            missing_count: 5,
            total_count: 100,
        };
        assert_eq!(feature.missing_percent(), 5.0);
    }

    #[test]
    fn test_dtype_icons() {
        assert_eq!(DataType::Numeric.icon(), "📊");
        assert_eq!(DataType::Categorical.icon(), "🏷️");
        assert_eq!(DataType::Binary.icon(), "🔀");
    }
}
