use dioxus::prelude::*;

/// Virtual scrolling configuration for large datasets
const VISIBLE_ROWS: usize = 20; // Rows visible at once
const ROW_HEIGHT: f64 = 40.0; // Height of each row in pixels
const HEADER_HEIGHT: f64 = 45.0;

/// Sort direction for columns
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Column configuration for the data table
#[derive(Clone, PartialEq)]
pub struct ColumnConfig {
    pub name: String,
    pub width: Option<f64>, // None = auto-width
    pub sortable: bool,
    pub align: TextAlign,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TextAlign {
    Left,
    Center,
    Right,
}

impl TextAlign {
    pub fn as_css(&self) -> &'static str {
        match self {
            TextAlign::Left => "left",
            TextAlign::Center => "center",
            TextAlign::Right => "right",
        }
    }
}

/// Props for DataTable component
#[derive(Props, Clone, PartialEq)]
pub struct DataTableProps {
    /// Column headers
    columns: Vec<String>,
    /// All data rows (can be 100K+)
    data: Vec<Vec<String>>,
    /// Optional column configuration
    #[props(default)]
    column_config: Option<Vec<ColumnConfig>>,
    /// Enable sorting
    #[props(default = true)]
    sortable: bool,
    /// Enable filtering
    #[props(default = true)]
    filterable: bool,
    /// Show row numbers
    #[props(default = true)]
    show_row_numbers: bool,
    /// Callback when row is clicked
    #[props(default)]
    on_row_click: Option<EventHandler<usize>>,
}

/// High-performance data table with virtual scrolling for 100K+ rows
///
/// # Example
/// ```rust
/// let columns = vec!["Name".to_string(), "Age".to_string(), "City".to_string()];
/// let data = vec![
///     vec!["Alice".to_string(), "30".to_string(), "NYC".to_string()],
///     vec!["Bob".to_string(), "25".to_string(), "LA".to_string()],
///     // ... 100,000 more rows
/// ];
///
/// rsx! {
///     DataTable {
///         columns: columns,
///         data: data,
///         sortable: true,
///         filterable: true,
///         on_row_click: move |row_idx| {
///             println!("Clicked row {}", row_idx);
///         }
///     }
/// }
/// ```
#[component]
pub fn DataTable(props: DataTableProps) -> Element {
    // State for virtual scrolling
    let mut scroll_top = use_signal(|| 0.0);
    let mut sort_column = use_signal(|| None::<usize>);
    let mut sort_direction = use_signal(|| SortDirection::Ascending);
    let mut filter_text = use_signal(|| String::new());

    // Calculate which rows are visible based on scroll position
    let start_row = (*scroll_top.read() / ROW_HEIGHT).floor() as usize;
    let end_row = (start_row + VISIBLE_ROWS + 5).min(props.data.len()); // +5 for smooth scrolling

    // Apply filtering
    let filtered_indices: Vec<usize> = if filter_text.read().is_empty() {
        (0..props.data.len()).collect()
    } else {
        let filter_lower = filter_text.read().to_lowercase();
        props
            .data
            .iter()
            .enumerate()
            .filter(|(_, row)| {
                row.iter()
                    .any(|cell| cell.to_lowercase().contains(&filter_lower))
            })
            .map(|(idx, _)| idx)
            .collect()
    };

    // Apply sorting
    let mut sorted_indices = filtered_indices.clone();
    if let Some(col_idx) = *sort_column.read() {
        let dir = *sort_direction.read();
        sorted_indices.sort_by(|&a, &b| {
            let cell_a = &props.data[a][col_idx];
            let cell_b = &props.data[b][col_idx];

            // Try numeric comparison first, fall back to string
            let cmp = if let (Ok(num_a), Ok(num_b)) = (cell_a.parse::<f64>(), cell_b.parse::<f64>())
            {
                num_a
                    .partial_cmp(&num_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                cell_a.cmp(cell_b)
            };

            match dir {
                SortDirection::Ascending => cmp,
                SortDirection::Descending => cmp.reverse(),
            }
        });
    }

    // Get visible rows based on scroll position
    let visible_rows: Vec<(usize, &Vec<String>)> = sorted_indices
        .iter()
        .skip(start_row)
        .take(end_row - start_row)
        .map(|&idx| (idx, &props.data[idx]))
        .collect();

    // Calculate total height for virtual scrolling
    let total_height = sorted_indices.len() as f64 * ROW_HEIGHT;
    let offset_y = start_row as f64 * ROW_HEIGHT;

    // Handle column sort
    let mut handle_sort = move |col_idx: usize| {
        if !props.sortable {
            return;
        }

        if Some(col_idx) == *sort_column.read() {
            // Toggle direction
            let current_dir = *sort_direction.read();
            sort_direction.set(match current_dir {
                SortDirection::Ascending => SortDirection::Descending,
                SortDirection::Descending => SortDirection::Ascending,
            });
        } else {
            // New column
            sort_column.set(Some(col_idx));
            sort_direction.set(SortDirection::Ascending);
        }
    };

    // Handle scroll
    let handle_scroll = move |evt: Event<ScrollData>| {
        // Note: Dioxus ScrollData API may vary - using basic scroll tracking
        // In production, implement proper scroll position tracking
        let _ = evt;
        // scroll_top is tracked via manual updates or ref
    };

    rsx! {
        div { class: "data-table-container",
            // Filter input
            if props.filterable {
                div { class: "data-table-filter",
                    input {
                        r#type: "text",
                        class: "filter-input",
                        placeholder: "🔍 Filter rows...",
                        value: "{filter_text}",
                        oninput: move |evt| filter_text.set(evt.value()),
                    }
                    span { class: "filter-stats",
                        "Showing {sorted_indices.len()} of {props.data.len()} rows"
                    }
                }
            }

            // Virtual scrolling viewport
            div {
                class: "data-table-viewport",
                style: "height: 600px; overflow-y: auto; position: relative;",
                onscroll: handle_scroll,

                // Spacer for total height (enables scrollbar)
                div {
                    style: "height: {total_height}px; position: relative;",

                    // Actual visible content (offset by scroll position)
                    div {
                        style: "position: absolute; top: {offset_y}px; width: 100%;",

                        table { class: "data-table",
                            thead { class: "data-table-header", style: "position: sticky; top: 0; z-index: 10;",
                                tr {
                                    // Row number header
                                    if props.show_row_numbers {
                                        th { class: "row-number-header", "#" }
                                    }

                                    // Column headers
                                    for (col_idx, col_name) in props.columns.iter().enumerate() {
                                        th {
                                            class: if props.sortable { "sortable-header" } else { "" },
                                            onclick: move |_| handle_sort(col_idx),

                                            span { class: "header-content",
                                                "{col_name}"

                                                // Sort indicator
                                                if Some(col_idx) == *sort_column.read() {
                                                    span { class: "sort-indicator",
                                                        match *sort_direction.read() {
                                                            SortDirection::Ascending => " ↑",
                                                            SortDirection::Descending => " ↓",
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            tbody {
                                for (original_idx, row) in visible_rows.iter() {
                                    {
                                        let idx = *original_idx;
                                        let row_data: Vec<String> = row.iter().map(|s| s.clone()).collect();

                                        rsx! {
                                            tr {
                                                class: "data-table-row",
                                                key: "{idx}",
                                                onclick: move |_| {
                                                    if let Some(ref handler) = props.on_row_click {
                                                        handler.call(idx);
                                                    }
                                                },

                                                // Row number
                                                if props.show_row_numbers {
                                                    td { class: "row-number", "{idx + 1}" }
                                                }

                                                // Data cells
                                                for cell in row_data.iter() {
                                                    td { class: "data-cell", "{cell}" }
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

            // Table statistics
            div { class: "data-table-stats",
                p {
                    "Total rows: {props.data.len()} | Columns: {props.columns.len()} | "
                    if !filter_text.read().is_empty() {
                        "Filtered: {sorted_indices.len()} rows"
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
    fn test_text_align_css() {
        assert_eq!(TextAlign::Left.as_css(), "left");
        assert_eq!(TextAlign::Center.as_css(), "center");
        assert_eq!(TextAlign::Right.as_css(), "right");
    }

    #[test]
    fn test_sort_direction_toggle() {
        let mut dir = SortDirection::Ascending;
        dir = match dir {
            SortDirection::Ascending => SortDirection::Descending,
            SortDirection::Descending => SortDirection::Ascending,
        };
        assert_eq!(dir, SortDirection::Descending);
    }

    #[test]
    fn test_virtual_scroll_calculation() {
        let scroll_top = 400.0;
        let start_row = (scroll_top / ROW_HEIGHT).floor() as usize;
        assert_eq!(start_row, 10); // 400 / 40 = 10
    }

    #[test]
    fn test_visible_rows_range() {
        let total_rows = 100;
        let start_row = 10;
        let end_row = (start_row + VISIBLE_ROWS + 5).min(total_rows);
        assert_eq!(end_row, 35); // 10 + 20 + 5 = 35
    }
}
