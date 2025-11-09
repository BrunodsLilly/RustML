use dioxus::prelude::*;
use loader::CsvDataset;

const MAX_FILE_SIZE: usize = 10 * 1024 * 1024; // 10 MB
const MAX_PREVIEW_ROWS: usize = 10;

#[derive(Clone, Debug)]
pub struct CsvPreview {
    pub headers: Vec<String>,
    pub sample_rows: Vec<Vec<String>>,
    pub total_rows: usize,
}

#[component]
pub fn CsvUploader(on_loaded: EventHandler<CsvDataset>) -> Element {
    let mut file_name = use_signal(|| String::new());
    let mut file_content = use_signal(|| Option::<String>::None);
    let mut preview = use_signal(|| Option::<CsvPreview>::None);
    let mut error_message = use_signal(|| Option::<String>::None);
    let mut loading = use_signal(|| false);
    let mut target_column = use_signal(|| String::new());

    let handle_upload = move |evt: Event<FormData>| async move {
        loading.set(true);
        error_message.set(None);
        preview.set(None);

        if let Some(file_engine) = evt.files() {
            let files = file_engine.files();

            if files.is_empty() {
                error_message.set(Some("No file selected".to_string()));
                loading.set(false);
                return;
            }

            let filename = files[0].clone();
            file_name.set(filename.clone());

            // Validate file extension
            if !filename.ends_with(".csv") {
                error_message.set(Some("Please upload a .csv file".to_string()));
                loading.set(false);
                return;
            }

            // Read file contents
            if let Some(contents) = file_engine.read_file_to_string(&filename).await {
                // Validate file size
                if contents.len() > MAX_FILE_SIZE {
                    error_message.set(Some(format!(
                        "File too large: {} MB (max 10 MB)",
                        contents.len() / 1024 / 1024
                    )));
                    loading.set(false);
                    return;
                }

                // Store content for later use
                file_content.set(Some(contents.clone()));

                // Parse CSV for preview
                match parse_csv_preview(&contents) {
                    Ok(prev) => {
                        // Auto-select last column as target
                        if let Some(last_header) = prev.headers.last() {
                            target_column.set(last_header.clone());
                        }
                        preview.set(Some(prev));
                    }
                    Err(e) => {
                        error_message.set(Some(format!("CSV parsing error: {}", e)));
                    }
                }
            } else {
                error_message.set(Some("Failed to read file".to_string()));
            }
        }

        loading.set(false);
    };

    let load_dataset = move |_| {
        if let Some(content) = file_content() {
            match CsvDataset::from_csv(&content, &target_column()) {
                Ok(dataset) => {
                    on_loaded.call(dataset);
                    error_message.set(None);
                }
                Err(e) => {
                    error_message.set(Some(format!("Dataset loading error: {}", e)));
                }
            }
        }
    };

    rsx! {
        section { class: "csv-uploader",
            h3 { "Upload CSV Dataset" }

            // File input
            div { class: "upload-controls",
                input {
                    r#type: "file",
                    accept: ".csv",
                    onchange: handle_upload,
                }

                if loading() {
                    span { class: "loading", "⏳ Processing..." }
                }
            }

            // Error display
            if let Some(err) = error_message() {
                div { class: "error-message",
                    "⚠️ {err}"
                }
            }

            // Preview table
            if let Some(prev) = preview() {
                div { class: "csv-preview",
                    h4 { "Preview: {file_name()} ({prev.total_rows} rows)" }

                    // Target column selector
                    div { class: "column-selector",
                        label { r#for: "target-column", "Target column: " }
                        select {
                            id: "target-column",
                            value: "{target_column()}",
                            onchange: move |evt| target_column.set(evt.value()),
                            for header in prev.headers.iter() {
                                option { value: "{header}", "{header}" }
                            }
                        }
                    }

                    // Data preview table
                    table { class: "preview-table",
                        thead {
                            tr {
                                for header in prev.headers.iter() {
                                    th { "{header}" }
                                }
                            }
                        }
                        tbody {
                            for row in prev.sample_rows.iter() {
                                tr {
                                    for cell in row.iter() {
                                        td { "{cell}" }
                                    }
                                }
                            }
                        }
                    }

                    if prev.total_rows > MAX_PREVIEW_ROWS {
                        p { class: "preview-note",
                            "Showing first {MAX_PREVIEW_ROWS} of {prev.total_rows} rows"
                        }
                    }

                    button {
                        class: "load-button",
                        onclick: load_dataset,
                        "Load Dataset"
                    }
                }
            }
        }
    }
}

fn parse_csv_preview(contents: &str) -> Result<CsvPreview, String> {
    use csv::ReaderBuilder;

    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(contents.as_bytes());

    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| e.to_string())?
        .iter()
        .map(|h| h.to_string())
        .collect();

    let mut sample_rows = Vec::new();
    let mut total_rows = 0;

    for (i, result) in reader.records().enumerate() {
        let record = result.map_err(|e| e.to_string())?;
        total_rows += 1;

        if i < MAX_PREVIEW_ROWS {
            let row: Vec<String> = record.iter().map(|s| s.to_string()).collect();
            sample_rows.push(row);
        }
    }

    Ok(CsvPreview {
        headers,
        sample_rows,
        total_rows,
    })
}
