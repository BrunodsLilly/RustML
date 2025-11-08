# CSV Integration Quick Reference

**Quick lookup for CSV processing in Dioxus WASM**

---

## 1. Dependencies

### Standard Approach (Recommended for MVP)
```toml
[dependencies]
csv = "1.3"
serde = { version = "1.0", features = ["derive"] }
```

### Zero-Allocation Approach (For Performance)
```toml
[dependencies]
serde-csv-core = "0.2"
serde = { version = "1.0", features = ["derive"] }
```

---

## 2. Dioxus File Upload

```rust
use dioxus::prelude::*;

#[component]
fn CsvUploader() -> Element {
    let mut csv_data = use_signal(|| String::new());

    rsx! {
        input {
            r#type: "file",
            accept: ".csv",
            multiple: false,
            onchange: move |evt: Event<FormData>| {
                async move {
                    if let Some(engine) = evt.files() {
                        if let Some(filename) = engine.files().first() {
                            if let Some(contents) = engine
                                .read_file_to_string(filename)
                                .await
                            {
                                csv_data.set(contents);
                            }
                        }
                    }
                }
            }
        }
        p { "Loaded {csv_data().len()} bytes" }
    }
}
```

---

## 3. CSV Parsing (Standard)

```rust
use csv::{Reader, ReaderBuilder};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct DataPoint {
    x: f64,
    y: f64,
    label: String,
}

fn parse_csv(data: &str) -> Result<Vec<DataPoint>, String> {
    let mut reader = Reader::from_reader(data.as_bytes());

    reader
        .deserialize()
        .collect::<Result<Vec<DataPoint>, _>>()
        .map_err(|e| format!("Parse error: {}", e))
}

// Custom delimiter (TSV)
fn parse_tsv(data: &str) -> Result<Vec<DataPoint>, String> {
    let mut reader = ReaderBuilder::new()
        .delimiter(b'\t')
        .from_reader(data.as_bytes());

    reader.deserialize().collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())
}
```

---

## 4. CSV Parsing (Zero-Allocation)

```rust
use serde_csv_core::Reader;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Point { x: f64, y: f64 }

fn parse_csv_fast(data: &str) -> Result<Vec<Point>, String> {
    let mut reader = Reader::<256>::new();
    let mut input = data.as_bytes();
    let mut scratch = [0u8; 1024];
    let mut results = Vec::new();

    loop {
        match reader.deserialize::<Point>(&input, &mut scratch) {
            Ok((record, n_bytes)) => {
                results.push(record);
                input = &input[n_bytes..];
            }
            Err(serde_csv_core::Error::EndOfCsvStream) => break,
            Err(e) => return Err(format!("{:?}", e)),
        }
    }

    Ok(results)
}
```

---

## 5. Convert to Matrix

```rust
use linear_algebra::Matrix;

fn to_matrix(points: Vec<DataPoint>) -> Result<Matrix<f64>, String> {
    let n = points.len();
    let data: Vec<f64> = points
        .iter()
        .flat_map(|p| vec![p.x, p.y])
        .collect();

    Matrix::from_vec(data, n, 2)
        .map_err(|e| format!("{:?}", e))
}
```

---

## 6. Complete Pipeline

```rust
use dioxus::prelude::*;

#[derive(Clone)]
struct TrainingData {
    features: Matrix<f64>,
    labels: Matrix<f64>,
}

#[component]
fn MlTrainer() -> Element {
    let mut data = use_signal(|| Option::<TrainingData>::None);
    let mut status = use_signal(|| String::from("No file"));

    rsx! {
        input {
            r#type: "file",
            accept: ".csv",
            onchange: move |evt: Event<FormData>| {
                async move {
                    status.set("Loading...".to_string());

                    if let Some(engine) = evt.files() {
                        if let Some(contents) = engine
                            .read_file_to_string(&engine.files()[0])
                            .await
                        {
                            match parse_and_process(&contents) {
                                Ok(training_data) => {
                                    data.set(Some(training_data));
                                    status.set("Ready".to_string());
                                }
                                Err(e) => status.set(format!("Error: {}", e)),
                            }
                        }
                    }
                }
            }
        }

        p { "{status()}" }

        {data().map(|d| rsx! {
            button {
                onclick: move |_| train_model(&d),
                "Train Model"
            }
        })}
    }
}

fn parse_and_process(csv: &str) -> Result<TrainingData, String> {
    let points: Vec<DataPoint> = parse_csv(csv)?;

    let features = to_matrix(points)?;
    let labels = extract_labels(&features)?;

    Ok(TrainingData { features, labels })
}
```

---

## 7. Error Handling

```rust
// ✅ GOOD: Don't panic, return Result
fn safe_parse(data: &str) -> Result<Vec<Point>, String> {
    let mut reader = Reader::from_reader(data.as_bytes());

    reader
        .deserialize::<Point>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Line {}: {}", e.position().unwrap().line(), e))
}

// ✅ GOOD: Validate data ranges
fn validate(points: &[Point]) -> Result<(), String> {
    for (i, p) in points.iter().enumerate() {
        if !p.x.is_finite() || !p.y.is_finite() {
            return Err(format!("Invalid value at row {}", i + 1));
        }
    }
    Ok(())
}
```

---

## 8. Performance Patterns

```rust
// Bounded memory (your circular buffer pattern)
const MAX_ROWS: usize = 10_000;

fn add_row(buffer: &mut Vec<Point>, row: Point) {
    if buffer.len() >= MAX_ROWS {
        buffer.remove(0);
    }
    buffer.push(row);
}

// Chunked processing (avoid blocking browser)
async fn parse_large_csv(data: String) -> Result<Vec<Point>, String> {
    let mut results = Vec::new();

    for chunk in data.lines().collect::<Vec<_>>().chunks(1000) {
        let chunk_data = chunk.join("\n");
        let parsed = parse_csv(&chunk_data)?;
        results.extend(parsed);

        // Yield to browser
        gloo_timers::future::sleep(Duration::from_millis(0)).await;
    }

    Ok(results)
}
```

---

## 9. Common Issues

### Issue: "unreachable executed" panic in WASM

```rust
// ❌ BAD
let value: f64 = row["price"].parse().unwrap();  // Panics!

// ✅ GOOD
let value: f64 = row.get("price")
    .ok_or("Missing price column")?
    .parse()
    .map_err(|_| "Invalid price")?;
```

### Issue: Large files freeze browser

```rust
// ❌ BAD
let all_data = parse_csv(&huge_file);  // Blocks for seconds

// ✅ GOOD
spawn(async move {
    let data = parse_csv_chunked(&huge_file).await;
    signal.set(data);
});
```

### Issue: Memory leaks during streaming

```rust
// ❌ BAD
loop {
    rows.push(parse_row());  // Unbounded growth!
}

// ✅ GOOD
const MAX_HISTORY: usize = 10_000;
if rows.len() >= MAX_HISTORY {
    rows.remove(0);  // Bounded circular buffer
}
rows.push(parse_row());
```

---

## 10. Testing Patterns

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_parse_basic() {
        let csv = "x,y\n1.0,2.0\n3.0,4.0";
        let result = parse_csv(csv).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].x, 1.0);
    }

    #[test]
    fn test_handle_errors() {
        let csv = "x,y\ninvalid,2.0";
        assert!(parse_csv(csv).is_err());
    }

    #[test]
    fn test_performance() {
        let csv = generate_csv(10_000);
        let start = std::time::Instant::now();
        let _ = parse_csv(&csv).unwrap();
        assert!(start.elapsed().as_millis() < 500);
    }
}
```

---

## 11. File Structure

```
cargo_workspace/
├── loader/                  # Add CSV parsing here
│   ├── Cargo.toml          # Add csv = "1.3"
│   └── src/
│       └── csv_loader.rs   # New module
│
└── web/
    ├── src/
    │   └── components/
    │       └── csv_upload.rs   # Upload component
    └── assets/
        └── sample.csv          # Test data
```

---

## 12. Useful Commands

```bash
# Test CSV loader
cargo test -p loader

# Run web app
cd web && dx serve

# Check WASM build
cd web && dx build --platform web

# Profile in browser
# Chrome DevTools → Performance tab → Record 10 seconds
```

---

## Resources

- Full docs: `/Users/brunodossantos/Code/brunoml/cargo_workspace/docs/CSV_INTEGRATION_RESEARCH.md`
- csv crate: https://docs.rs/csv/latest/csv/
- serde-csv-core: https://docs.rs/serde-csv-core/
- Dioxus file upload: https://medium.com/@mikecode/dioxus-59-handle-file-input-2e0c9a913880
