use dioxus::prelude::*;

/// Severity level for data quality issues
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

impl Severity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Severity::Info => "Info",
            Severity::Warning => "Warning",
            Severity::Error => "Error",
            Severity::Critical => "Critical",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Severity::Info => "ℹ️",
            Severity::Warning => "⚠️",
            Severity::Error => "❌",
            Severity::Critical => "🚨",
        }
    }

    pub fn color(&self) -> &'static str {
        match self {
            Severity::Info => "#3498db",
            Severity::Warning => "#f39c12",
            Severity::Error => "#e74c3c",
            Severity::Critical => "#c0392b",
        }
    }
}

/// Data quality issue
#[derive(Clone, Debug, PartialEq)]
pub struct QualityIssue {
    pub severity: Severity,
    pub column: String,
    pub issue_type: IssueType,
    pub description: String,
    pub affected_rows: usize,
    pub recommendation: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum IssueType {
    MissingValues,
    Outliers,
    Duplicates,
    InvalidValues,
    HighCardinality,
    LowVariance,
    Imbalance,
    InconsistentFormat,
}

impl IssueType {
    pub fn as_str(&self) -> &'static str {
        match self {
            IssueType::MissingValues => "Missing Values",
            IssueType::Outliers => "Outliers Detected",
            IssueType::Duplicates => "Duplicate Rows",
            IssueType::InvalidValues => "Invalid Values",
            IssueType::HighCardinality => "High Cardinality",
            IssueType::LowVariance => "Low Variance",
            IssueType::Imbalance => "Class Imbalance",
            IssueType::InconsistentFormat => "Inconsistent Format",
        }
    }
}

/// Analyze data quality and detect issues
pub fn analyze_quality(
    columns: &[String],
    data: &[Vec<String>],
) -> Vec<QualityIssue> {
    let mut issues = Vec::new();

    if data.is_empty() {
        return issues;
    }

    let n_rows = data.len();

    for (col_idx, col_name) in columns.iter().enumerate() {
        let column_data: Vec<&str> = data.iter().map(|row| row[col_idx].as_str()).collect();

        // Check for missing values
        let missing_count = column_data.iter().filter(|s| s.trim().is_empty()).count();
        if missing_count > 0 {
            let missing_pct = (missing_count as f64 / n_rows as f64) * 100.0;
            let severity = if missing_pct > 50.0 {
                Severity::Critical
            } else if missing_pct > 20.0 {
                Severity::Error
            } else if missing_pct > 5.0 {
                Severity::Warning
            } else {
                Severity::Info
            };

            issues.push(QualityIssue {
                severity,
                column: col_name.clone(),
                issue_type: IssueType::MissingValues,
                description: format!("{:.1}% of values are missing ({} rows)", missing_pct, missing_count),
                affected_rows: missing_count,
                recommendation: if missing_pct > 50.0 {
                    "Consider dropping this column or imputing with domain knowledge".to_string()
                } else if missing_pct > 20.0 {
                    "Use imputation (mean/median for numeric, mode for categorical)".to_string()
                } else {
                    "Small amount of missing data, safe to drop rows or use simple imputation".to_string()
                },
            });
        }

        // Check for numeric columns
        let numeric_values: Vec<f64> = column_data
            .iter()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();

        if !numeric_values.is_empty() && numeric_values.len() > n_rows / 2 {
            // Outlier detection using IQR method
            let mut sorted = numeric_values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let q1 = percentile(&sorted, 25.0);
            let q3 = percentile(&sorted, 75.0);
            let iqr = q3 - q1;
            let lower_bound = q1 - 1.5 * iqr;
            let upper_bound = q3 + 1.5 * iqr;

            let outlier_count = numeric_values
                .iter()
                .filter(|&&v| v < lower_bound || v > upper_bound)
                .count();

            if outlier_count > 0 {
                let outlier_pct = (outlier_count as f64 / numeric_values.len() as f64) * 100.0;
                if outlier_pct > 1.0 {
                    issues.push(QualityIssue {
                        severity: if outlier_pct > 10.0 { Severity::Warning } else { Severity::Info },
                        column: col_name.clone(),
                        issue_type: IssueType::Outliers,
                        description: format!("{:.1}% of values are outliers ({} values)", outlier_pct, outlier_count),
                        affected_rows: outlier_count,
                        recommendation: "Consider robust scaling or capping outliers at percentile thresholds".to_string(),
                    });
                }
            }

            // Low variance detection
            if numeric_values.len() > 10 {
                let mean = numeric_values.iter().sum::<f64>() / numeric_values.len() as f64;
                let variance = numeric_values
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / numeric_values.len() as f64;
                let std_dev = variance.sqrt();

                if std_dev < 0.01 {
                    issues.push(QualityIssue {
                        severity: Severity::Warning,
                        column: col_name.clone(),
                        issue_type: IssueType::LowVariance,
                        description: format!("Very low variance (std={:.6})", std_dev),
                        affected_rows: 0,
                        recommendation: "Consider removing this feature as it provides little information".to_string(),
                    });
                }
            }
        }

        // Check for high cardinality in categorical columns
        let unique_values: std::collections::HashSet<&str> =
            column_data.iter().filter(|s| !s.trim().is_empty()).copied().collect();
        let cardinality = unique_values.len();
        let cardinality_ratio = cardinality as f64 / n_rows as f64;

        if cardinality > 50 && cardinality_ratio > 0.5 {
            issues.push(QualityIssue {
                severity: Severity::Info,
                column: col_name.clone(),
                issue_type: IssueType::HighCardinality,
                description: format!("{} unique values ({:.1}% of rows)", cardinality, cardinality_ratio * 100.0),
                affected_rows: cardinality,
                recommendation: "Consider grouping rare categories or using target encoding".to_string(),
            });
        }
    }

    // Check for duplicate rows
    let unique_rows: std::collections::HashSet<Vec<String>> =
        data.iter().cloned().collect();
    let duplicate_count = n_rows - unique_rows.len();

    if duplicate_count > 0 {
        let dup_pct = (duplicate_count as f64 / n_rows as f64) * 100.0;
        issues.push(QualityIssue {
            severity: if dup_pct > 10.0 { Severity::Warning } else { Severity::Info },
            column: "Dataset".to_string(),
            issue_type: IssueType::Duplicates,
            description: format!("{:.1}% duplicate rows ({} rows)", dup_pct, duplicate_count),
            affected_rows: duplicate_count,
            recommendation: "Remove duplicate rows using drop_duplicates()".to_string(),
        });
    }

    // Sort by severity
    issues.sort_by(|a, b| b.severity.cmp(&a.severity));

    issues
}

fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let n = sorted_values.len();
    let index = (p / 100.0) * (n - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let fraction = index - lower as f64;

    if lower == upper {
        sorted_values[lower]
    } else {
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    }
}

/// Props for DataQuality component
#[derive(Props, Clone, PartialEq)]
pub struct DataQualityProps {
    /// Quality issues detected
    issues: Vec<QualityIssue>,
    /// Show detailed recommendations
    #[props(default = true)]
    show_recommendations: bool,
    /// Group issues by column
    #[props(default = false)]
    group_by_column: bool,
}

/// Data quality dashboard showing issues and recommendations
///
/// # Example
/// ```rust
/// let issues = analyze_quality(&columns, &data);
///
/// rsx! {
///     DataQuality {
///         issues: issues,
///         show_recommendations: true,
///     }
/// }
/// ```
#[component]
pub fn DataQuality(props: DataQualityProps) -> Element {
    // Count issues by severity
    let critical = props.issues.iter().filter(|i| i.severity == Severity::Critical).count();
    let errors = props.issues.iter().filter(|i| i.severity == Severity::Error).count();
    let warnings = props.issues.iter().filter(|i| i.severity == Severity::Warning).count();
    let info = props.issues.iter().filter(|i| i.severity == Severity::Info).count();

    // Overall quality score (100 - weighted penalties)
    let quality_score = 100.0 -
        (critical as f64 * 25.0 +
         errors as f64 * 15.0 +
         warnings as f64 * 5.0 +
         info as f64 * 1.0).min(100.0);

    rsx! {
        div { class: "data-quality-container",
            h3 { class: "data-quality-title", "🔍 Data Quality Report" }

            // Quality Score
            div { class: "quality-score-card",
                div { class: "score-gauge",
                    div {
                        class: "score-circle",
                        style: "background: conic-gradient(
                            {score_color(quality_score)} {quality_score}%,
                            #e9ecef {quality_score}%
                        )",

                        div { class: "score-inner",
                            span { class: "score-value", "{quality_score:.0}" }
                            span { class: "score-label", "Quality Score" }
                        }
                    }
                }

                div { class: "score-breakdown",
                    div { class: "breakdown-item critical",
                        span { "🚨 Critical" }
                        span { class: "breakdown-count", "{critical}" }
                    }
                    div { class: "breakdown-item error",
                        span { "❌ Errors" }
                        span { class: "breakdown-count", "{errors}" }
                    }
                    div { class: "breakdown-item warning",
                        span { "⚠️ Warnings" }
                        span { class: "breakdown-count", "{warnings}" }
                    }
                    div { class: "breakdown-item info",
                        span { "ℹ️ Info" }
                        span { class: "breakdown-count", "{info}" }
                    }
                }
            }

            // Issues List
            if props.issues.is_empty() {
                div { class: "no-issues",
                    "✅ No data quality issues detected! Your data looks great."
                }
            } else {
                div { class: "issues-list",
                    for issue in props.issues.iter() {
                        IssueCard {
                            issue: issue.clone(),
                            show_recommendation: props.show_recommendations,
                        }
                    }
                }
            }

            // Quick Actions
            if !props.issues.is_empty() {
                div { class: "quality-actions",
                    h4 { "🛠️ Recommended Actions" }
                    ul {
                        if critical > 0 || errors > 0 {
                            li { "Address critical issues before training models" }
                        }
                        if warnings > 0 {
                            li { "Review warnings and apply appropriate data cleaning" }
                        }
                        li { "Export quality report for documentation" }
                        li { "Rerun analysis after cleaning to verify improvements" }
                    }
                }
            }
        }
    }
}

fn score_color(score: f64) -> &'static str {
    if score >= 90.0 {
        "#51cf66" // Green
    } else if score >= 70.0 {
        "#ffd43b" // Yellow
    } else if score >= 50.0 {
        "#ff922b" // Orange
    } else {
        "#ff6b6b" // Red
    }
}

/// Props for IssueCard component
#[derive(Props, Clone, PartialEq)]
pub struct IssueCardProps {
    issue: QualityIssue,
    show_recommendation: bool,
}

/// Individual issue card
#[component]
pub fn IssueCard(props: IssueCardProps) -> Element {
    let severity_class = match props.issue.severity {
        Severity::Critical => "issue-card critical",
        Severity::Error => "issue-card error",
        Severity::Warning => "issue-card warning",
        Severity::Info => "issue-card info",
    };

    rsx! {
        div { class: severity_class,
            div { class: "issue-header",
                span { class: "issue-icon", "{props.issue.severity.icon()}" }
                div { class: "issue-title",
                    h5 { "{props.issue.issue_type.as_str()}" }
                    span { class: "issue-column", "Column: {props.issue.column}" }
                }
                span {
                    class: "issue-badge",
                    style: "background-color: {props.issue.severity.color()}",
                    "{props.issue.severity.as_str()}"
                }
            }

            div { class: "issue-description",
                p { "{props.issue.description}" }
                if props.issue.affected_rows > 0 {
                    p { class: "affected-rows", "Affects {props.issue.affected_rows} rows" }
                }
            }

            if props.show_recommendation {
                div { class: "issue-recommendation",
                    strong { "💡 Recommendation: " }
                    "{props.issue.recommendation}"
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Critical > Severity::Error);
        assert!(Severity::Error > Severity::Warning);
        assert!(Severity::Warning > Severity::Info);
    }

    #[test]
    fn test_analyze_missing_values() {
        let columns = vec!["A".to_string(), "B".to_string()];
        let data = vec![
            vec!["1".to_string(), "".to_string()],
            vec!["2".to_string(), "x".to_string()],
        ];

        let issues = analyze_quality(&columns, &data);
        assert!(issues.iter().any(|i| i.issue_type == IssueType::MissingValues));
    }

    #[test]
    fn test_analyze_duplicates() {
        let columns = vec!["A".to_string()];
        let data = vec![
            vec!["1".to_string()],
            vec!["1".to_string()],
            vec!["2".to_string()],
        ];

        let issues = analyze_quality(&columns, &data);
        assert!(issues.iter().any(|i| i.issue_type == IssueType::Duplicates));
    }

    #[test]
    fn test_quality_score() {
        let score = 85.0;
        assert_eq!(score_color(score), "#ffd43b");
    }
}
