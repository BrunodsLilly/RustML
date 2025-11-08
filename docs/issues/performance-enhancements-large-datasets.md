# feat: Performance Enhancements for Large Dataset Handling

## Overview

Optimize the CSV upload and training pipeline to handle production-scale datasets (100k+ rows, 50+ features) with streaming parsing, Web Worker offloading, progressive rendering, and memory optimization. Prove that Rust + WASM can match or exceed Python for large-scale ML in the browser.

## Problem Statement

The current CSV upload and training implementation (from PR #3) works well for small datasets but has performance limitations:
- **Memory:** Entire CSV loaded into memory before processing
- **Blocking:** CSV parsing blocks main thread (UI freeze)
- **Rendering:** Large preview tables cause jank
- **Training:** Long training freezes UI for minutes
- **Scaling:** 100k rows × 50 features → crashes or extreme slowness

**Impact:** Users can't work with realistic datasets, limiting real-world applicability. The platform's promise of "native-speed ML in browser" isn't proven at scale.

## Proposed Solution

Implement production-grade performance optimizations while maintaining the client-side, zero-backend philosophy:

1. **Streaming CSV Parsing** - Process data incrementally, not all at once
2. **Web Worker Offloading** - Move parsing and training off main thread
3. **Progressive Rendering** - Virtual scrolling for large tables
4. **Memory Optimization** - Compact data structures, bounded buffers
5. **Performance Monitoring** - Real-time metrics dashboard

## Technical Approach

### Architecture

**New Modules:**
```
loader/src/
├── streaming_csv.rs      (NEW - incremental parsing)
├── compact_storage.rs    (NEW - memory-efficient formats)
└── lib.rs                (MODIFY - streaming APIs)

web/src/
├── workers/
│   ├── csv_parser.worker.js  (NEW - Web Worker for parsing)
│   └── trainer.worker.js     (NEW - Web Worker for training)
├── components/
│   ├── virtual_table.rs      (NEW - progressive rendering)
│   ├── performance_monitor.rs (NEW - metrics dashboard)
│   └── showcase.rs           (MODIFY - async loading)
└── wasm_bridge.rs            (NEW - Worker<->WASM communication)
```

**Data Flow:**
```
Large CSV File
    ↓
Web Worker (csv_parser) → Streaming Parser → Chunks
    ↓
Main Thread ← Progress Updates (10% ... 100%)
    ↓
Compact Matrix Storage (f32 instead of f64)
    ↓
Web Worker (trainer) → Training Loop
    ↓
Main Thread ← Loss Updates (every 100 iter)
```

### Implementation Phases

#### Phase 1: Streaming CSV Parsing (Week 1, 16-20 hours)

**Tasks:**
- [ ] Implement streaming parser in `loader/src/streaming_csv.rs`
  - Parse CSV in chunks (configurable chunk size)
  - Yield after each chunk to avoid blocking
  - Validate and accumulate into Matrix incrementally
  - Handle errors mid-stream gracefully
- [ ] Add progress callbacks
  - Report bytes processed / total bytes
  - Estimate time remaining
  - Cancel parsing mid-stream
- [ ] Optimize memory allocation
  - Pre-allocate Matrix if row count known (from `wc -l` equivalent)
  - Use Vec::with_capacity to avoid reallocations
  - Stream directly to target format (no intermediate strings)
- [ ] Add benchmarks
  - Compare streaming vs batch for 10k, 100k, 1M rows
  - Measure peak memory usage
  - Validate no memory leaks
- [ ] Add tests for streaming
- [ ] Add E2E test: upload 100k row CSV, verify no UI freeze

**Files to Create:**
```rust
// loader/src/streaming_csv.rs
use std::io::BufRead;

pub struct StreamingCsvParser {
    chunk_size: usize,
    buffer: Vec<Vec<f64>>,
    headers: Vec<String>,
    target_column: String,
}

impl StreamingCsvParser {
    pub fn new(target_column: String, chunk_size: usize) -> Self {
        Self {
            chunk_size,
            buffer: Vec::with_capacity(chunk_size),
            headers: vec![],
            target_column,
        }
    }

    pub async fn parse_stream<R: BufRead>(
        &mut self,
        reader: R,
        progress_callback: impl Fn(usize, usize),  // (bytes_processed, total_bytes)
    ) -> Result<CsvDataset, CsvError> {
        let mut csv_reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(reader);

        // Read headers
        self.headers = csv_reader.headers()?.iter()
            .map(|s| s.to_string())
            .collect();

        let target_idx = self.headers.iter()
            .position(|h| h == &self.target_column)
            .ok_or(CsvError::TargetNotFound(self.target_column.clone()))?;

        let mut all_features = vec![];
        let mut all_targets = vec![];
        let mut bytes_processed = 0usize;

        // Process records in chunks
        for (row_num, result) in csv_reader.records().enumerate() {
            let record = result?;

            // Parse row
            let mut row_features = vec![];
            let mut target = 0.0;

            for (col_idx, value) in record.iter().enumerate() {
                let parsed: f64 = value.parse()
                    .map_err(|_| CsvError::ParseError {
                        column: self.headers[col_idx].clone(),
                        line: row_num + 2,
                        value: value.to_string(),
                    })?;

                if col_idx == target_idx {
                    target = parsed;
                } else {
                    row_features.push(parsed);
                }
            }

            all_features.push(row_features);
            all_targets.push(target);

            // Update progress every chunk
            if (row_num + 1) % self.chunk_size == 0 {
                bytes_processed += record.as_slice().len();
                progress_callback(bytes_processed, usize::MAX);  // Total unknown in stream

                // Yield to event loop (WASM-safe)
                TimeoutFuture::new(0).await;
            }
        }

        // Convert to Matrix
        let n_samples = all_features.len();
        let n_features = all_features[0].len();

        let feature_data: Vec<f64> = all_features.into_iter().flatten().collect();
        let features = Matrix::from_vec(feature_data, n_samples, n_features)?;

        Ok(CsvDataset {
            features,
            targets: all_targets,
            feature_names: self.headers.iter()
                .enumerate()
                .filter(|&(i, _)| i != target_idx)
                .map(|(_, name)| name.clone())
                .collect(),
            num_samples: n_samples,
        })
    }
}

// Helper for yielding to event loop in WASM
use async_std::task;
use std::time::Duration;

struct TimeoutFuture(Duration);

impl TimeoutFuture {
    fn new(millis: u64) -> Self {
        Self(Duration::from_millis(millis))
    }
}

impl std::future::Future for TimeoutFuture {
    type Output = ();

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context) -> std::task::Poll<()> {
        // Use wasm_bindgen_futures::spawn_local or similar
        task::sleep(self.0);
        std::task::Poll::Ready(())
    }
}
```

**Web Integration:**
```rust
// web/src/components/csv_upload.rs (modifications)
let parse_csv_streaming = use_resource(move || async move {
    let mut parser = StreamingCsvParser::new(target_column(), 1000);

    let progress = |bytes, total| {
        parsing_progress.set(Some((bytes, total)));
    };

    parser.parse_stream(file_reader, progress).await
});
```

**Success Criteria:**
- ✅ Parses 100k rows without blocking UI
- ✅ Peak memory <200 MB for 100k × 20 dataset
- ✅ Progress updates every 1000 rows (10-30 FPS)
- ✅ Can cancel mid-parse without memory leak
- ✅ 2-3x faster than batch parsing for large files

#### Phase 2: Web Worker Offloading (Week 2, 20-24 hours)

**Tasks:**
- [ ] Create Web Worker wrapper for WASM
  - Compile WASM module loadable in Worker
  - Message passing protocol (JSON or binary)
  - Shared memory for large transfers (optional)
  - Error propagation to main thread
- [ ] Implement CSV parsing worker
  - Receives file via transferable object
  - Runs streaming parser
  - Sends parsed Matrix back
  - Reports progress via postMessage
- [ ] Implement training worker
  - Receives dataset and hyperparameters
  - Runs training loop
  - Sends loss updates periodically
  - Returns trained model when complete
- [ ] Add worker lifecycle management
  - Terminate workers on unmount
  - Handle worker crashes gracefully
  - Warm start workers on app load
- [ ] Add benchmarks comparing Worker vs main thread
- [ ] Add tests for Worker communication
- [ ] Add E2E test: upload 50k CSV, verify UI responsive during parse

**Files to Create:**
```javascript
// web/src/workers/csv_parser.worker.js
import init, { StreamingCsvParser } from '../pkg/web.js';

let wasm_initialized = false;

self.onmessage = async function(e) {
    if (!wasm_initialized) {
        await init();
        wasm_initialized = true;
    }

    const { type, payload } = e.data;

    switch (type) {
        case 'PARSE_CSV':
            const { file_content, target_column } = payload;

            try {
                // Progress callback
                const onProgress = (bytes, total) => {
                    self.postMessage({
                        type: 'PROGRESS',
                        payload: { bytes, total },
                    });
                };

                // Parse in WASM
                const parser = new StreamingCsvParser(target_column, 1000);
                const dataset = await parser.parse_stream(file_content, onProgress);

                // Send result back
                self.postMessage({
                    type: 'PARSE_COMPLETE',
                    payload: dataset,
                });
            } catch (error) {
                self.postMessage({
                    type: 'PARSE_ERROR',
                    payload: error.toString(),
                });
            }
            break;

        case 'CANCEL':
            // Terminate parsing (implement cancellation token)
            break;
    }
};

// web/src/workers/trainer.worker.js
import init, { LinearRegressor } from '../pkg/web.js';

let wasm_initialized = false;

self.onmessage = async function(e) {
    if (!wasm_initialized) {
        await init();
        wasm_initialized = true;
    }

    const { type, payload } = e.data;

    switch (type) {
        case 'TRAIN':
            const { features, targets, learning_rate, iterations } = payload;

            try {
                const model = new LinearRegressor(learning_rate);

                // Training loop with progress
                for (let i = 0; i < iterations; i++) {
                    model.step(features, targets);

                    // Send progress every 100 iterations
                    if (i % 100 === 0) {
                        const loss = model.get_current_loss();
                        self.postMessage({
                            type: 'TRAINING_PROGRESS',
                            payload: { iteration: i, loss },
                        });
                    }
                }

                // Send trained model
                self.postMessage({
                    type: 'TRAINING_COMPLETE',
                    payload: model.serialize(),
                });
            } catch (error) {
                self.postMessage({
                    type: 'TRAINING_ERROR',
                    payload: error.toString(),
                });
            }
            break;
    }
};
```

**Rust Bridge:**
```rust
// web/src/wasm_bridge.rs
use wasm_bindgen::prelude::*;
use web_sys::{Worker, WorkerOptions};

#[wasm_bindgen]
pub struct WorkerPool {
    workers: Vec<Worker>,
}

#[wasm_bindgen]
impl WorkerPool {
    pub fn new(num_workers: usize) -> Self {
        let workers = (0..num_workers)
            .map(|_| {
                let opts = WorkerOptions::new();
                Worker::new_with_options("./csv_parser.worker.js", &opts).unwrap()
            })
            .collect();

        Self { workers }
    }

    pub fn parse_csv(&self, file_content: String, target_column: String) -> js_sys::Promise {
        let worker = &self.workers[0];

        let message = js_sys::Object::new();
        js_sys::Reflect::set(&message, &"type".into(), &"PARSE_CSV".into()).unwrap();

        let payload = js_sys::Object::new();
        js_sys::Reflect::set(&payload, &"file_content".into(), &file_content.into()).unwrap();
        js_sys::Reflect::set(&payload, &"target_column".into(), &target_column.into()).unwrap();
        js_sys::Reflect::set(&message, &"payload".into(), &payload).unwrap();

        worker.post_message(&message).unwrap();

        // Return promise that resolves when worker sends PARSE_COMPLETE
        // ... implementation using JsFuture
        unimplemented!()
    }
}
```

**Success Criteria:**
- ✅ UI stays responsive during 100k row parse
- ✅ Training 1M iterations doesn't freeze UI
- ✅ Worker startup <200ms
- ✅ Message passing overhead <10ms per update
- ✅ Worker crashes don't crash main app

#### Phase 3: Progressive Rendering & Virtual Scrolling (Week 3, 16-20 hours)

**Tasks:**
- [ ] Implement virtual table component
  - Render only visible rows (viewport)
  - Dynamic row height support
  - Smooth scrolling with pre-rendering buffer
  - Handle 100k+ rows without DOM bloat
- [ ] Add column virtualization (optional)
  - Render only visible columns for wide datasets
  - Horizontal scrolling
- [ ] Optimize preview rendering
  - Debounce scroll updates
  - Use CSS transforms for smooth scroll
  - Lazy load cell content
- [ ] Add performance monitoring
  - FPS counter
  - Render time per frame
  - Memory usage graph
- [ ] Add tests for virtual scrolling
- [ ] Add E2E test: scroll through 50k row table, verify smooth 60 FPS

**Files to Create:**
```rust
// web/src/components/virtual_table.rs
use dioxus::prelude::*;

#[component]
pub fn VirtualTable(
    data: Vec<Vec<String>>,  // All rows
    headers: Vec<String>,
    row_height: f64,          // Fixed height per row in px
) -> Element {
    let mut scroll_top = use_signal(|| 0.0);
    let viewport_height = 600.0;  // Visible area height
    let buffer_rows = 10;          // Pre-render buffer

    // Calculate visible range
    let start_idx = (scroll_top() / row_height).floor() as usize;
    let end_idx = ((scroll_top() + viewport_height) / row_height).ceil() as usize;

    let visible_start = start_idx.saturating_sub(buffer_rows);
    let visible_end = (end_idx + buffer_rows).min(data.len());

    rsx! {
        div {
            class: "virtual-table-container",
            style: "height: {viewport_height}px; overflow-y: scroll;",
            onscroll: move |evt| {
                scroll_top.set(evt.target().scroll_top() as f64);
            },

            // Spacer for total height
            div {
                style: "height: {data.len() as f64 * row_height}px; position: relative;",

                // Table header (sticky)
                div {
                    class: "table-header",
                    style: "position: sticky; top: 0; z-index: 10;",
                    {headers.iter().map(|h| rsx! {
                        div { class: "header-cell", "{h}" }
                    })}
                }

                // Visible rows only
                div {
                    style: "position: absolute; top: {visible_start as f64 * row_height}px;",
                    {data[visible_start..visible_end].iter().enumerate().map(|(i, row)| {
                        let actual_idx = visible_start + i;
                        rsx! {
                            div {
                                key: "{actual_idx}",
                                class: "table-row",
                                style: "height: {row_height}px;",
                                {row.iter().map(|cell| rsx! {
                                    div { class: "table-cell", "{cell}" }
                                })}
                            }
                        }
                    })}
                }
            }
        }
    }
}
```

**Performance Monitoring:**
```rust
// web/src/components/performance_monitor.rs
#[component]
pub fn PerformanceMonitor() -> Element {
    let mut fps = use_signal(|| 0.0);
    let mut memory_mb = use_signal(|| 0.0);

    use_effect(move || {
        // FPS tracking
        let mut frame_times = vec![];
        let mut last_frame = window().performance().unwrap().now();

        let closure = Closure::wrap(Box::new(move || {
            let now = window().performance().unwrap().now();
            frame_times.push(now - last_frame);
            last_frame = now;

            if frame_times.len() > 60 {
                frame_times.remove(0);
            }

            let avg_frame_time = frame_times.iter().sum::<f64>() / frame_times.len() as f64;
            fps.set(1000.0 / avg_frame_time);
        }) as Box<dyn FnMut()>);

        window().request_animation_frame(closure.as_ref().unchecked_ref());
        closure.forget();

        // Memory tracking (if available)
        if let Some(memory) = window().performance().unwrap().memory() {
            memory_mb.set(memory.used_js_heap_size() as f64 / 1_048_576.0);
        }
    });

    rsx! {
        div { class: "performance-monitor",
            div { "FPS: {fps():.1}" }
            div { "Memory: {memory_mb():.1} MB" }
        }
    }
}
```

**Success Criteria:**
- ✅ Smooth 60 FPS scrolling through 100k rows
- ✅ Initial render <100ms for any dataset size
- ✅ Memory usage O(visible rows), not O(total rows)
- ✅ No jank on rapid scrolling
- ✅ Works on mobile devices

#### Phase 4: Memory Optimization & Benchmarking (Week 4, 12-16 hours)

**Tasks:**
- [ ] Implement compact data storage
  - Use f32 instead of f64 where precision isn't critical
  - Columnar storage for better cache locality
  - Compression for sparse datasets (optional)
  - Memory pooling for matrix allocations
- [ ] Add dataset size limits and warnings
  - Warn at 50k rows or 50 features
  - Block at 1M rows (prevent crashes)
  - Show estimated memory usage
- [ ] Create comprehensive benchmarks
  - Test matrix: 1k, 10k, 100k, 1M rows × 10, 50, 100 features
  - Measure: parse time, training time, memory, FPS
  - Compare: batch vs streaming, main vs Worker
  - Generate performance report
- [ ] Add performance regression tests
  - CI checks for performance degradation
  - Alert if >20% slower than baseline
- [ ] Document performance characteristics
- [ ] Add E2E stress test: 100k × 50 dataset, full workflow

**Files to Create:**
```rust
// loader/src/compact_storage.rs
use half::f16;  // 16-bit float for extreme compression

#[derive(Clone)]
pub enum CompactMatrix {
    F64(Matrix<f64>),     // Full precision
    F32(Matrix<f32>),     // Half memory, sufficient for most ML
    F16(Matrix<f16>),     // Quarter memory, lossy but usable
}

impl CompactMatrix {
    pub fn from_precision(data: Vec<f64>, rows: usize, cols: usize, precision: Precision) -> Self {
        match precision {
            Precision::Full => CompactMatrix::F64(Matrix::from_vec(data, rows, cols).unwrap()),
            Precision::Half => {
                let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                CompactMatrix::F32(Matrix::from_vec(f32_data, rows, cols).unwrap())
            }
            Precision::Quarter => {
                let f16_data: Vec<f16> = data.iter().map(|&x| f16::from_f64(x)).collect();
                CompactMatrix::F16(Matrix::from_vec(f16_data, rows, cols).unwrap())
            }
        }
    }

    pub fn memory_bytes(&self) -> usize {
        match self {
            CompactMatrix::F64(m) => m.rows * m.cols * 8,
            CompactMatrix::F32(m) => m.rows * m.cols * 4,
            CompactMatrix::F16(m) => m.rows * m.cols * 2,
        }
    }

    pub fn to_f64(&self) -> Matrix<f64> {
        match self {
            CompactMatrix::F64(m) => m.clone(),
            CompactMatrix::F32(m) => {
                let data: Vec<f64> = m.data.iter().map(|&x| x as f64).collect();
                Matrix::from_vec(data, m.rows, m.cols).unwrap()
            }
            CompactMatrix::F16(m) => {
                let data: Vec<f64> = m.data.iter().map(|&x| x.to_f64()).collect();
                Matrix::from_vec(data, m.rows, m.cols).unwrap()
            }
        }
    }
}

pub enum Precision {
    Full,     // f64 - 8 bytes/value
    Half,     // f32 - 4 bytes/value
    Quarter,  // f16 - 2 bytes/value
}
```

**Benchmarking Suite:**
```rust
// linear_regression/benches/large_datasets.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_training_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_scale");

    for n_samples in [1_000, 10_000, 100_000].iter() {
        for n_features in [10, 50, 100].iter() {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}x{}", n_samples, n_features)),
                &(n_samples, n_features),
                |b, &(n, f)| {
                    let X = generate_random_matrix(*n, *f);
                    let y = generate_random_vector(*n);
                    let mut model = LinearRegressor::new(0.01);

                    b.iter(|| {
                        model.fit(black_box(&X), black_box(&y), 100);
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_training_scale);
criterion_main!(benches);
```

**Success Criteria:**
- ✅ f32 storage uses 50% memory of f64 with <1% accuracy loss
- ✅ Handles 100k × 50 dataset in <500 MB total memory
- ✅ Training throughput >10k samples/sec
- ✅ No memory leaks after 100 training runs
- ✅ Performance report documents all characteristics

## Alternative Approaches Considered

### 1. Server-Side Processing (Rejected)
**Pros:** Easier to scale, no browser limits
**Cons:** Breaks privacy promise, requires backend
**Decision:** Optimize client-side to match server performance

### 2. AssemblyScript Instead of Rust (Rejected)
**Pros:** Simpler Worker integration
**Cons:** Worse performance than Rust, limited ML ecosystem
**Decision:** Stick with Rust, invest in Worker bridge

### 3. SharedArrayBuffer for Zero-Copy (Future)
**Pros:** Eliminates data copying between threads
**Cons:** Requires COOP/COEP headers, security concerns
**Decision:** Use Transferables for now, add SharedArrayBuffer later

## Acceptance Criteria

### Functional Requirements

- [ ] **Streaming Parsing**
  - Processes 100k rows without blocking UI
  - Progress updates every 1000 rows
  - Cancellable mid-stream
  - Memory efficient (no full copy)

- [ ] **Web Workers**
  - Parsing runs in Worker
  - Training runs in Worker
  - UI stays responsive
  - Handles Worker crashes

- [ ] **Virtual Scrolling**
  - Renders 100k row table smoothly
  - 60 FPS scrolling
  - Minimal memory footprint
  - Works on mobile

- [ ] **Memory Optimization**
  - f32 storage option
  - Bounded buffers
  - Handles 100k × 50 in <500 MB
  - No leaks

### Non-Functional Requirements

- [ ] **Performance**
  - Parse 100k rows: <5s
  - Train 100k samples: <30s
  - Virtual scroll: 60 FPS
  - Worker startup: <200ms

- [ ] **Scalability**
  - Supports up to 1M rows
  - Up to 100 features
  - Graceful degradation beyond limits
  - Clear error messages at scale

- [ ] **Reliability**
  - No crashes for large datasets
  - Handles edge cases (empty, 1 row, etc.)
  - Recoverable errors
  - Predictable memory usage

### Quality Gates

- [ ] **Benchmarks**
  - Comprehensive benchmark suite
  - CI integration
  - Performance regression tests
  - Documented baselines

- [ ] **Stress Tests**
  - 100k × 50 full workflow
  - 1M row parsing
  - 1 hour continuous training
  - Memory leak detection

## Success Metrics

**Technical:**
- 10x throughput improvement for large datasets
- 50% memory reduction with f32
- 60 FPS maintained at all dataset sizes
- Zero crashes for datasets within limits

**User Impact:**
- 30%+ upload >10k row datasets
- Average dataset size increases 5x
- Positive feedback on "handles real data"
- No complaints about freezing/crashing

## Dependencies & Prerequisites

**Required:**
- ✅ CSV parsing (existing)
- ✅ Linear regression (existing)
- ✅ WASM compilation (existing)

**New:**
- Web Workers support in build
- `half` crate for f16
- `criterion` for benchmarking
- Performance profiling tools

## Risk Analysis & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Worker overhead negates gains | High | Medium | Benchmark early, use Transferables, batch updates |
| WASM memory limits (2 GB) | High | Low | Monitor usage, block at 80%, compress data |
| Virtual scroll bugs | Medium | Medium | Extensive testing, gradual rollout |
| f32 precision loss | Medium | Low | A/B test accuracy, allow f64 toggle |

## Resource Requirements

**Time:** 4 weeks (64-80 hours)
- Week 1: Streaming (16-20h)
- Week 2: Workers (20-24h)
- Week 3: Virtual scroll (16-20h)
- Week 4: Optimization (12-16h)

**Team:** 1 senior developer (Rust + WASM + perf experience)

## Future Considerations

- GPU acceleration via WebGPU
- SharedArrayBuffer for zero-copy
- IndexedDB for dataset persistence
- Distributed training across tabs
- Quantization for model compression

## References

### Internal
- CSV Parser: `loader/src/csv_loader.rs`
- Matrix Ops: `linear_algebra/src/lib.rs`
- Existing Performance: `neural_network/src/optimizer.rs`

### External
- [Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API)
- [Transferable Objects](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferable_objects)
- [Virtual Scrolling](https://developer.mozilla.org/en-US/docs/Web/CSS/overflow)
- [WASM Performance](https://rustwasm.github.io/docs/book/reference/code-size.html)

---

**Labels:** `enhancement`, `performance`, `optimization`, `large-scale`

**Estimated Effort:** Large (4 weeks)

**Priority:** P2 (Critical for Production Use)
