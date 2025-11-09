# Educational ML Visualization Interface - Best Practices Research

**Research Date:** November 8, 2025
**Purpose:** Identify best practices for building educational, self-documenting ML visualization interfaces for the Rust+WASM ML Playground

---

## Executive Summary

This research synthesizes patterns from leading educational ML platforms (TensorFlow Playground, Distill.pub, Observable, VisuAlgo) and UI/UX best practices to guide the development of revolutionary ML visualization interfaces. Key findings emphasize **progressive disclosure**, **real-time visual feedback**, **consistent color coding**, and **WASM-specific error handling patterns**.

**Golden Rules Identified:**
1. **Immediate Visual Feedback** - Updates within 100ms feel instantaneous
2. **Orange/Blue Color Convention** - Industry standard for negative/positive values
3. **Progressive Disclosure** - Hide complexity, reveal on demand
4. **Video-like Controls** - Play/pause/step for algorithm execution
5. **Zero-Setup Learning** - No installation, runs in browser
6. **Result Types Over Panics** - WASM panics are unrecoverable

---

## 1. Educational ML Platforms: Proven Patterns

### 1.1 TensorFlow Playground (playground.tensorflow.org)

**Status:** Active, 10+ years of refinement
**Authority:** Google's official educational tool, millions of users

#### Key Design Patterns

**A. Hierarchical Control Organization**
```
Layout Structure:
├─ Top: Primary Controls (Epoch, Learning Rate, Activation)
├─ Left: Data Selection & Features
├─ Center: Live Visualization (Network Graph)
└─ Right: Performance Metrics (Loss Curves)
```

**Implementation Insight:**
- Group related controls in labeled sections
- Default to 80% use case, hide advanced features
- Use sliders for continuous parameters (0.00001–10 range with logarithmic scale)
- Use dropdowns for categorical choices (activation functions, problem types)

**B. Consistent Color Encoding**
```
Color Language (Applied Universally):
- Blue = Positive values
- Orange = Negative values
- Intensity = Confidence/magnitude
- Line thickness = Weight magnitude
```

**Why It Works:** Reduces cognitive load - users learn once, apply everywhere

**C. Real-Time Visualization**
```
Update Loop (60 FPS target):
1. User clicks "Train"
2. Every iteration updates:
   - Decision boundary background
   - Neuron connection weights (animated thickness/color)
   - Loss curves (training + test)
   - Epoch counter
3. User can pause, resume, or step through iterations
```

**Performance Note:** Achieves 60 FPS with Canvas rendering + Web Workers

**D. Progressive Disclosure**
```
Visibility Layers:
Level 0 (Always Visible):
  - Play/Pause/Reset controls
  - Current epoch and loss
  - Data visualization

Level 1 (Expandable Sections):
  - "What Is a Neural Network?" explanation
  - Advanced feature engineering
  - Regularization options

Level 2 (External Links):
  - Detailed mathematical foundations
  - Research papers
```

**Educational Impact:** Beginners aren't overwhelmed, experts can dive deep

#### Code Example: Logarithmic Slider Range
```rust
// TensorFlow Playground uses logarithmic scale for learning rate
const MIN_EXP: f64 = -5.0;  // 10^-5
const MAX_EXP: f64 = 1.0;   // 10^1

fn slider_to_learning_rate(slider_value: f64) -> f64 {
    // slider_value: 0.0 to 1.0
    let exponent = MIN_EXP + slider_value * (MAX_EXP - MIN_EXP);
    10.0_f64.powf(exponent)
}

// Example: slider=0.5 → exponent=-2.0 → learning_rate=0.01
```

**Why Logarithmic?** Captures meaningful differences across orders of magnitude

---

### 1.2 Distill.pub - Interactive Research Articles

**Status:** Archive (2016-2021), but patterns remain gold standard
**Authority:** Leading ML research communication platform

#### Key Design Patterns

**A. Layered Visual Explanations**
```
Explanation Hierarchy:
1. Icon-based visual (neuron symbols, layer diagrams)
2. Natural language description
3. Mathematical notation (inline, not blocking)
4. Live interactive demonstration
5. Code snippet (expandable)
```

**Example Structure:**
```markdown
## Feature Visualization

[Icon: Neural network layer]

Instead of showing a neuron as abstract math, we can **visualize what it detects**.
Below, slide to see different feature activations:

[Interactive slider with live image updates]

Mathematically, this is optimizing: `layern[:,:,z]` where...

<details>
  <summary>View implementation code</summary>
  ```python
  # Optimization code here
  ```
</details>
```

**B. Comparative Visualization**
```
Side-by-Side Pattern (Before/After):

┌─────────────────────┬─────────────────────┐
│   Without Diversity │   With Diversity    │
│   Term              │   Term              │
├─────────────────────┼─────────────────────┤
│ [Single blurry     │ [Four distinct      │
│  visualization]     │  visualizations]    │
│                     │                     │
│ Interpretation:     │ Interpretation:     │
│ Unclear pattern    │ Multiple facets     │
│                     │ revealed            │
└─────────────────────┴─────────────────────┘
```

**Why It Works:** Shows concrete impact, not abstract claims

**C. Progressive Complexity Scaffolding**
```
Article Structure (Problem-First Narrative):
1. Show FAILURE example first (motivates solution)
   → "The Enemy of Feature Visualization"
   → Concrete example of high-frequency artifacts

2. Explain WHY problem occurs
   → Simple intuition before math

3. Present SOLUTION incrementally
   → Basic regularization → Advanced techniques

4. Prove with RESULTS
   → Side-by-side comparisons with/without technique
```

**Educational Philosophy:** Motivate before teaching, prove before claiming

**D. Embedded Interactive Diagrams**
```html
<!-- Distill pattern: Custom HTML elements -->
<d-slider bind="interpolation_t" min="0" max="1" step="0.01"></d-slider>

<!-- Result updates in real-time as user drags -->
<d-figure id="interpolation-viz">
  <!-- SVG visualization here -->
</d-figure>
```

**Technical Implementation:**
- Custom Web Components for reusable interactions
- Reactive data binding (slider changes → visualization updates)
- No page reloads, smooth 60 FPS animations

#### Best Practices Extracted

1. **Visual Proofs Over Claims:** Show results, don't just describe
2. **Inline Interactivity:** Diagrams appear mid-paragraph, maintain flow
3. **Unified Visual Language:** Consistent layout for related concepts (comparison grids)
4. **Expandable Technical Depth:** Math and code behind `<details>` tags

---

### 1.3 Observable - Reactive Notebooks

**Status:** Active, recently released Observable 2.0
**Authority:** Created by D3.js creator Mike Bostock

#### Key Design Patterns

**A. Reactive Cell Execution**
```javascript
// Cells automatically re-run when dependencies change
viewof k = Inputs.range([2, 10], {value: 3, step: 1, label: "k clusters"})

// This cell auto-updates when k changes
kmeans_result = runKMeans(data, k)

// Visualization auto-updates when kmeans_result changes
Plot.dot(data, {
  x: "feature1",
  y: "feature2",
  fill: d => kmeans_result.labels[d.index]
})
```

**Why Revolutionary:** Users see cause-and-effect immediately (change k → clusters update)

**B. Lightweight Input Components**
```javascript
// Observable Inputs - built-in UI widgets
viewof selected_features = Inputs.checkbox(
  ["age", "income", "education"],
  {label: "Features to include:", value: ["age", "income"]}
)

viewof learning_rate = Inputs.range(
  [0.001, 1],
  {value: 0.01, step: 0.001, label: "Learning Rate", transform: Math.log}
)

viewof algorithm = Inputs.radio(
  ["K-Means", "PCA", "LogReg"],
  {label: "Algorithm:", value: "K-Means"}
)
```

**UI Pattern Recommendation:** Declarative inputs with immediate visual feedback

**C. Two-Step Interactivity**
```
Pattern: Input → Visualization Linking

Step 1: Create Input
  viewof parameter = Inputs.slider(...)

Step 2: Connect to Visualization
  // Use 'parameter' in any downstream cell
  // Automatic reactivity - no event handlers needed
```

**Rust+WASM Equivalent:**
```rust
// Dioxus reactive pattern
let mut k_clusters = use_signal(|| 3);

// Input slider
input {
    r#type: "range",
    min: "2",
    max: "10",
    value: "{k_clusters}",
    oninput: move |evt| k_clusters.set(evt.value().parse().unwrap_or(3))
}

// Auto-updates when k_clusters changes
rsx! {
    KMeansVisualization { k: *k_clusters.read() }
}
```

**D. Observable Plot - Declarative Viz**
```javascript
// High-level grammar of graphics
Plot.plot({
  marks: [
    Plot.dot(data, {x: "x", y: "y", fill: "cluster"}),
    Plot.line(centroids, {x: "x", y: "y", stroke: "red", strokeWidth: 3})
  ],
  color: {legend: true}
})
```

**Key Insight:** Declarative specs reduce boilerplate (vs. imperative D3.js)

#### Best Practices Extracted

1. **Reactive Programming Model:** Change input → visualization auto-updates
2. **Built-in Inputs Library:** Standard widgets (sliders, checkboxes, dropdowns)
3. **Collaborative Features:** Real-time multiplayer, version history, forking
4. **Zero Setup:** Runs entirely in browser, no local environment

---

### 1.4 VisuAlgo - Algorithm Step-Through Platform

**Status:** Active since 2011, 24 visualization modules
**Authority:** Used by National University of Singapore CS courses

#### Key Design Patterns

**A. User-Controllable Execution**
```
Video-Like Controls:
┌─────────────────────────────────────────┐
│ [◀◀] [◀] [▶] [▶▶] [Speed: 2x ▼]      │
│  First Prev Play Next  Last            │
└─────────────────────────────────────────┘

Speed Options: 0.5x, 1x, 2x, 4x
```

**Why It Works:** Users control pacing, can pause to reflect on each step

**Implementation Pattern:**
```rust
struct AlgorithmPlayer {
    snapshots: Vec<AlgorithmState>,  // Pre-computed steps
    current_index: usize,
    speed_multiplier: f64,
    is_playing: bool,
}

impl AlgorithmPlayer {
    fn step_forward(&mut self) {
        if self.current_index < self.snapshots.len() - 1 {
            self.current_index += 1;
        }
    }

    fn step_backward(&mut self) {
        if self.current_index > 0 {
            self.current_index -= 1;
        }
    }

    fn play(&mut self) {
        // Advance frame every (1000ms / speed_multiplier)
        self.is_playing = true;
    }
}
```

**B. Snapshot-Based Animation**
```
Architecture:
1. Pre-compute all algorithm states (snapshots)
2. Store snapshots in array
3. Render current snapshot on each frame
4. Advance index on play, or user can seek manually
```

**Example K-Means Snapshots:**
```rust
struct KMeansSnapshot {
    iteration: usize,
    centroids: Vec<Point>,
    assignments: Vec<usize>,
    inertia: f64,
    highlighted_changes: Vec<usize>,  // Which points changed cluster
}

fn compute_snapshots(data: &Matrix<f64>, k: usize, max_iter: usize) -> Vec<KMeansSnapshot> {
    let mut snapshots = vec![];
    let mut kmeans = KMeans::new(k, max_iter, 1e-4, Some(42));

    // Modify fit() to record state after each iteration
    for iteration in 0..max_iter {
        // ... K-Means iteration logic ...

        snapshots.push(KMeansSnapshot {
            iteration,
            centroids: kmeans.centroids.clone(),
            assignments: kmeans.labels.clone(),
            inertia: kmeans.inertia,
            highlighted_changes: compute_changes(&prev_labels, &kmeans.labels),
        });
    }

    snapshots
}
```

**C. Algorithm Input Customization**
```
User Flow:
1. Select algorithm (e.g., "Quick Sort")
2. Choose input mode:
   - Random array (size N)
   - Manually enter values: [5, 2, 8, 1, 9]
   - Load from examples (nearly sorted, reverse sorted)
3. Click "Visualize"
4. Step through execution
```

**Educational Value:** Users see algorithm behavior on their own data

**D. Built-in Quiz System**
```
Quiz Pattern:
1. User watches algorithm visualization
2. System generates random question:
   "What is the time complexity of Quick Sort in the worst case?"
   A) O(n)
   B) O(n log n)
   C) O(n²)
   D) O(2ⁿ)

3. User answers, gets immediate feedback
4. Explanation with visualization link
```

**Implementation:**
```rust
struct QuizQuestion {
    question_text: String,
    options: Vec<String>,
    correct_answer: usize,
    explanation: String,
    related_visualization: String,  // URL to specific algorithm step
}

fn generate_kmeans_quiz() -> QuizQuestion {
    QuizQuestion {
        question_text: "When does K-Means converge?".to_string(),
        options: vec![
            "When all centroids are at origin".to_string(),
            "When cluster assignments stop changing".to_string(),
            "After exactly k iterations".to_string(),
            "When inertia reaches zero".to_string(),
        ],
        correct_answer: 1,
        explanation: "K-Means converges when cluster assignments stabilize...".to_string(),
        related_visualization: "/kmeans?step=convergence".to_string(),
    }
}
```

#### Best Practices Extracted

1. **Snapshot Architecture:** Pre-compute all states for instant seeking
2. **Variable Speed Playback:** 0.5x to 4x range accommodates different learning speeds
3. **User Input Support:** Let users test with their own data
4. **Quiz Integration:** Test understanding immediately after visualization
5. **SVG Over Canvas:** VisuAlgo migrated from Canvas to SVG for better scalability

---

### 1.5 Python Tutor - Code Execution Visualizer

**Status:** Active since 2010, 20M+ users in 180 countries
**Authority:** Most widely-used program visualization tool for CS education

#### Key Design Patterns

**A. Split-Panel Layout**
```
┌─────────────────────────┬─────────────────────────┐
│   Code (Left)           │  Memory State (Right)   │
├─────────────────────────┼─────────────────────────┤
│ 1  def kmeans(data, k): │  Frames:                │
│ 2    centroids = []     │  ┌─────────────────┐   │
│ 3►   for i in range(k): │  │ kmeans          │   │
│ 4      ...              │  │  data: [...]    │   │
│                         │  │  k: 3           │   │
│                         │  │  centroids: ↓   │   │
│                         │  └─────────│───────┘   │
│                         │            ▼            │
│                         │  Objects:               │
│                         │  list: [empty]          │
│                         │                         │
│ [◀ Prev] [Next ▶]     │  Print Output:          │
│ Step 3 of 42           │  (none yet)             │
└─────────────────────────┴─────────────────────────┘
```

**Why It Works:** Direct correspondence - "this line creates this variable"

**B. Visual Encoding of Memory**
```
Variable Representations:
- Primitives (int, float, bool): Inline values
  age: 25

- Objects (list, dict): Arrows to heap
  data: ─────→ [1.5, 2.3, 4.1]
               │
               └─ index markers [0] [1] [2]

- References: Arrows show aliasing
  centroids: ───┐
                 ├──→ [0.5, 0.5]
  new_cent:  ───┘
```

**Educational Value:** Visualizes pointer confusion, a major beginner pain point

**C. Backward/Forward Stepping**
```
Navigation Controls:
- First: Jump to line 1
- Previous: Step backward one line
- Next: Step forward one line
- Last: Jump to final state
- Slider: Seek to arbitrary step

Current State: "Step 14 of 87"
```

**Performance Consideration:**
- Pre-compute all execution states (like VisuAlgo snapshots)
- Store minimal diff between states for memory efficiency
- Max execution steps: typically capped at ~1000 to prevent memory issues

**D. Live Edit Mode**
```
User Flow:
1. User steps through code visualization
2. Spots bug at step 23
3. Clicks "Edit Code"
4. Fixes bug inline
5. Visualization re-runs automatically
6. User continues from step 1
```

**Revolutionary Aspect:** Tight feedback loop (visualize → fix → re-visualize)

#### Best Practices Extracted

1. **Split-Panel Layout:** Code on left, state on right (industry standard)
2. **Arrow-Based Memory Visualization:** Show object references explicitly
3. **Bidirectional Stepping:** Forward AND backward navigation
4. **Live Edit Capability:** Fix bugs mid-visualization
5. **Step Counter:** "Step X of Y" provides progress context
6. **Print Output Display:** Show console output separately from memory state

---

## 2. Data Exploration UI Patterns

### 2.1 CSV Data Preview Best Practices

**Sources:** Adobe Experience Platform, Justinmind, Andrew Coyle (Medium), Enterprise Data Table Patterns

#### Key Patterns

**A. Pagination Strategy**
```
Optimal Defaults (Research-Backed):
- Default: 25 rows per page (balance between density and performance)
- Options: [10, 25, 50, 100] rows per page
- Client-side pagination: Datasets < 1,000 rows
- Server-side pagination: Datasets ≥ 1,000 rows
```

**Why 25 Rows?** Studies show optimal balance between:
- Enough data to see patterns (statistical significance)
- Fast rendering (under 100ms load time)
- Comfortable scrolling distance

**B. Virtual Scrolling for Large Datasets**
```rust
// Pattern: Only render visible rows + buffer
struct VirtualTable {
    total_rows: usize,           // 100,000 total
    visible_rows: usize,         // 20 visible at once
    buffer_rows: usize,          // 10 above + 10 below (smooth scrolling)
    viewport_start_index: usize, // Current scroll position
}

impl VirtualTable {
    fn render_range(&self) -> Range<usize> {
        let start = self.viewport_start_index.saturating_sub(self.buffer_rows);
        let end = (self.viewport_start_index + self.visible_rows + self.buffer_rows)
            .min(self.total_rows);
        start..end
    }
}
```

**Performance Impact:** 100K rows renders in ~50ms vs. 5+ seconds for full table

**C. Display Density Control**
```
User Preference (Toggle):
┌──────────────────────────────┐
│ Density: [Compact] [Standard] [Comfortable] │
└──────────────────────────────┘

Compact:
  - Row height: 32px
  - Font size: 12px
  - Padding: 4px
  - Best for: Power users, large datasets

Standard:
  - Row height: 48px
  - Font size: 14px
  - Padding: 8px
  - Best for: General use

Comfortable:
  - Row height: 64px
  - Font size: 16px
  - Padding: 12px
  - Best for: Accessibility, presentations
```

**Why It Matters:** Different tasks need different density (scanning vs. detailed reading)

**D. Sorting & Filtering**
```
Multi-Level Sorting:
Primary: ↓ Department (descending)
Secondary: ↑ Revenue (ascending)
Tertiary: ↓ Date (descending)

Real-Time Search Filter:
┌─────────────────────────────┐
│ 🔍 Filter columns...        │
└─────────────────────────────┘
As user types "eng" → Only shows rows with "eng" in any column
```

**Implementation Pattern:**
```rust
struct TableFilter {
    search_term: String,
    sort_columns: Vec<(usize, SortDirection)>,  // Column index + direction
}

impl TableFilter {
    fn apply(&self, data: &Matrix<f64>, column_names: &[String]) -> Vec<usize> {
        let mut row_indices: Vec<usize> = (0..data.rows()).collect();

        // 1. Filter by search term
        if !self.search_term.is_empty() {
            row_indices.retain(|&i| {
                // Check if any column contains search_term
                (0..data.cols()).any(|j| {
                    // Convert to string and check
                    data.get(i, j).unwrap().to_string().contains(&self.search_term)
                })
            });
        }

        // 2. Apply multi-level sorting
        row_indices.sort_by(|&a, &b| {
            for (col_idx, direction) in &self.sort_columns {
                let val_a = data.get(a, *col_idx).unwrap();
                let val_b = data.get(b, *col_idx).unwrap();

                let cmp = match direction {
                    SortDirection::Ascending => val_a.partial_cmp(val_b).unwrap(),
                    SortDirection::Descending => val_b.partial_cmp(val_a).unwrap(),
                };

                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });

        row_indices
    }
}
```

**E. Column Management**
```
Responsive Column Handling:

Desktop (1920px+):
  [✓] All 15 columns visible
  [↔] Horizontal scrolling disabled

Tablet (768-1920px):
  [✓] 8 key columns visible
  [☐] 7 hidden columns (show/hide menu)
  [↔] Horizontal scroll enabled

Mobile (< 768px):
  [✓] 3 essential columns visible
  [☐] 12 hidden (tap row → detail view)
  [⬇] Vertical stacking of rows
```

**Fixed Column Pattern:**
```css
/* Fix first 2 columns (ID, Name) during horizontal scroll */
.table-column-0, .table-column-1 {
  position: sticky;
  left: 0;
  z-index: 10;
  background: white;
  box-shadow: 2px 0 4px rgba(0,0,0,0.1);
}
```

**F. Mobile Responsiveness**
```
Mobile Strategy: Card-Based Layout

Desktop Table:
┌────┬──────────┬─────┬─────────┐
│ ID │ Name     │ Age │ Income  │
├────┼──────────┼─────┼─────────┤
│ 1  │ Alice    │ 25  │ 50000   │
└────┴──────────┴─────┴─────────┘

Mobile Cards:
┌─────────────────────────┐
│ Alice (#1)              │
│ Age: 25                 │
│ Income: $50,000         │
│ [View Details →]       │
└─────────────────────────┘
```

**Implementation:**
```rust
// Responsive component
rsx! {
    div { class: "table-container",
        // Desktop: full table
        table { class: "desktop-table",
            thead { /* ... */ }
            tbody { /* render all columns */ }
        }

        // Mobile: card layout
        div { class: "mobile-cards",
            for row in filtered_data {
                div { class: "card",
                    h3 { "{row.name} (#{row.id})" }
                    p { "Age: {row.age}" }
                    p { "Income: ${row.income}" }
                    button { "View Details →" }
                }
            }
        }
    }
}
```

**G. Data Type Inference**

**Sources:** Research papers (Semi-automatic Column Type Inference for CSV), Python libraries (pandas, tableschema), Snowflake

**Automatic Type Detection Algorithm:**
```rust
#[derive(Debug, Clone, PartialEq)]
enum ColumnType {
    Integer,
    Float,
    Boolean,
    Date,
    Categorical(Vec<String>),  // Enum with <= 10 unique values
    Text,
}

fn infer_column_type(values: &[String]) -> ColumnType {
    // Strategy: Sample rows with longest, shortest, and first 50 values
    let sample = create_representative_sample(values, 50);

    // 1. Try parsing as boolean
    if sample.iter().all(|v| matches!(v.to_lowercase().as_str(), "true" | "false" | "0" | "1")) {
        return ColumnType::Boolean;
    }

    // 2. Try parsing as integer
    if sample.iter().all(|v| v.parse::<i64>().is_ok()) {
        return ColumnType::Integer;
    }

    // 3. Try parsing as float
    if sample.iter().all(|v| v.parse::<f64>().is_ok()) {
        return ColumnType::Float;
    }

    // 4. Try parsing as date
    if sample.iter().all(|v| parse_date(v).is_ok()) {
        return ColumnType::Date;
    }

    // 5. Check if categorical (low cardinality)
    let unique_values: HashSet<_> = sample.iter().cloned().collect();
    if unique_values.len() <= 10 && unique_values.len() < sample.len() / 2 {
        return ColumnType::Categorical(unique_values.into_iter().collect());
    }

    // 6. Default to text
    ColumnType::Text
}

fn create_representative_sample(values: &[String], max_samples: usize) -> Vec<String> {
    let mut sample = vec![];

    // Include longest value (often catches edge cases)
    if let Some(longest) = values.iter().max_by_key(|v| v.len()) {
        sample.push(longest.clone());
    }

    // Include shortest value
    if let Some(shortest) = values.iter().min_by_key(|v| v.len()) {
        sample.push(shortest.clone());
    }

    // Include first 50 non-empty values
    for val in values.iter().filter(|v| !v.is_empty()).take(max_samples) {
        sample.push(val.clone());
    }

    sample
}
```

**Visual Type Indicators:**
```
CSV Preview with Type Badges:

┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Age [123]    │ Name [abc]   │ Income [1.2] │ Active [✓]  │
│ Integer      │ Text         │ Float        │ Boolean     │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 25           │ Alice        │ 50000.00     │ true        │
│ 30           │ Bob          │ 65000.00     │ false       │
└──────────────┴──────────────┴──────────────┴──────────────┘

Legend:
[123] = Integer (blue badge)
[abc] = Text (gray badge)
[1.2] = Float (green badge)
[✓]   = Boolean (purple badge)
[📅]  = Date (orange badge)
```

#### Best Practices Summary

1. **Pagination:** 25 rows default, client-side < 1K rows, server-side ≥ 1K
2. **Virtual Scrolling:** Render only visible rows + buffer for large datasets
3. **Display Density:** Offer 3 levels (compact, standard, comfortable)
4. **Multi-Level Sorting:** Support sorting by multiple columns
5. **Real-Time Search:** Filter-as-you-type across all columns
6. **Fixed Columns:** Sticky ID/name columns during horizontal scroll
7. **Mobile Cards:** Switch to card layout on small screens
8. **Auto Type Inference:** Detect column types from sample data
9. **Visual Type Indicators:** Show data types with badges/icons

---

### 2.2 Variable/Feature Selection Interfaces

**Sources:** Northstar (MIT drag-and-drop analytics), PatternFly, Drag-Drop UX guides

#### Key Patterns

**A. Checkbox Multi-Select**
```
Pattern: Simple Checkbox List

Features Available:
☑ Age
☑ Income
☐ Education
☑ Employment Status
☐ Marital Status

[Run K-Means with 3 selected features]
```

**When to Use:** 5-20 features, all visible at once

**Implementation:**
```rust
let mut selected_features = use_signal(|| vec![true, true, false, true, false]);

div { class: "feature-selector",
    h3 { "Select Features for K-Means:" }

    for (i, feature_name) in feature_names.iter().enumerate() {
        label { class: "checkbox-label",
            input {
                r#type: "checkbox",
                checked: selected_features.read()[i],
                onchange: move |evt| {
                    let mut features = selected_features.write();
                    features[i] = evt.checked();
                }
            }
            span { "{feature_name}" }
        }
    }

    button {
        disabled: selected_features.read().iter().filter(|&&x| x).count() < 2,
        onclick: move |_| run_kmeans(),
        "Run K-Means ({selected_count} features)"
    }
}
```

**B. Drag-and-Drop Interface**

**Pattern:** Northstar-Style (MIT Research)
```
┌──────────────────────┬──────────────────────┐
│  Available Features  │  Selected Features   │
├──────────────────────┼──────────────────────┤
│  Drag from here:     │  Drop here:          │
│  ┌────────────────┐  │  ┌────────────────┐ │
│  │ Age            │  │  │ Income         │ │
│  │ Education      │  │  │ Employment     │ │
│  │ Marital Status │  │  │                │ │
│  └────────────────┘  │  └────────────────┘ │
│                      │                      │
│                      │  [Clear All]         │
└──────────────────────┴──────────────────────┘

[Analyze with selected features →]
```

**When to Use:** 20+ features, need visual grouping/organization

**Implementation:**
```rust
use dioxus::prelude::*;

fn FeatureDragDrop(cx: Scope) -> Element {
    let available_features = use_state(cx, || vec!["Age", "Education", "Marital Status"]);
    let selected_features = use_state(cx, || vec!["Income", "Employment"]);
    let dragged_feature = use_state(cx, || None::<String>);

    cx.render(rsx! {
        div { class: "drag-drop-container",
            // Available features panel
            div {
                class: "drop-zone",
                ondragover: move |evt| evt.prevent_default(),
                ondrop: move |evt| {
                    if let Some(feature) = dragged_feature.get() {
                        // Move from selected back to available
                        selected_features.modify(|f| f.retain(|x| x != &feature));
                        available_features.modify(|f| f.push(feature.clone()));
                    }
                },

                h3 { "Available Features" }
                for feature in available_features.get() {
                    div {
                        class: "draggable-feature",
                        draggable: "true",
                        ondragstart: move |_| dragged_feature.set(Some(feature.clone())),
                        "{feature}"
                    }
                }
            }

            // Selected features panel
            div {
                class: "drop-zone selected",
                ondragover: move |evt| evt.prevent_default(),
                ondrop: move |evt| {
                    if let Some(feature) = dragged_feature.get() {
                        // Move from available to selected
                        available_features.modify(|f| f.retain(|x| x != &feature));
                        selected_features.modify(|f| f.push(feature.clone()));
                    }
                },

                h3 { "Selected Features ({selected_features.len()})" }
                for feature in selected_features.get() {
                    div {
                        class: "draggable-feature selected",
                        draggable: "true",
                        ondragstart: move |_| dragged_feature.set(Some(feature.clone())),
                        "{feature}"
                    }
                }
            }
        }

        button {
            disabled: selected_features.len() < 2,
            "Analyze with {selected_features.len()} features →"
        }
    })
}
```

**C. Search + Filter Pattern**
```
┌─────────────────────────────────────┐
│ 🔍 Search features...               │
└─────────────────────────────────────┘

Filter by type:
[All] [Numeric] [Categorical] [Date]

Results (8 matching):
☑ Age (numeric)
☑ Income (numeric)
☐ Employment Status (categorical)
☐ Education Level (categorical)
...
```

**When to Use:** 50+ features, need to narrow down options

**D. Multi-Select with Drag Handle**

**PatternFly Pattern:**
```
┌──────────────────────────────────┐
│ ☐ [≡] Age                       │
│ ☑ [≡] Income          [Remove]  │
│ ☑ [≡] Employment      [Remove]  │
│ ☐ [≡] Education                 │
└──────────────────────────────────┘

Legend:
☐/☑ = Checkbox (multi-select)
[≡] = Drag handle (reorder)
```

**Implementation Detail:**
```rust
// Drag handle should be LEFT of checkbox for accessibility
// Allows keyboard users to tab through checkboxes without triggering drag
```

#### Best Practices Summary

1. **Checkbox List:** Best for ≤ 20 features, simple selection
2. **Drag-and-Drop:** Best for 20+ features, visual organization
3. **Search/Filter:** Required for 50+ features
4. **Multi-Select Drag:** Combines checkbox + reordering (most powerful)
5. **Visual Feedback:** Highlight drop zones, show selected count
6. **Keyboard Accessibility:** Support arrow keys, space to select, enter to confirm
7. **Minimum Selection:** Disable "Run" button if < 2 features selected (ML requirement)

---

## 3. Algorithm Step-Through Patterns

**Synthesis from:** VisuAlgo, Python Tutor, Algorithm Visualizer projects

### 3.1 Execution Control Patterns

**A. Video-Like Playback Controls**
```
Standard Layout:
┌────────────────────────────────────────────────┐
│  [|◀◀] [◀] [▶||] [▶] [▶▶|]   Speed: [2x ▼]  │
│  First  Prev Pause Next Last                  │
│                                                │
│  Progress: [=========>         ] 45%          │
│  Step: 127 / 280                              │
│                                                │
│  ⏱ Elapsed: 2.3s | Remaining: ~2.8s           │
└────────────────────────────────────────────────┘
```

**Speed Options (Industry Standard):**
- 0.5x (slow, for learning)
- 1x (normal)
- 2x (fast)
- 4x (very fast, for skipping)

**Keyboard Shortcuts:**
- Space: Play/Pause
- Left Arrow: Previous step
- Right Arrow: Next step
- Home: Jump to start
- End: Jump to end

**Implementation:**
```rust
struct StepPlayer {
    steps: Vec<AlgorithmSnapshot>,
    current_index: usize,
    speed: f64,              // Multiplier: 0.5, 1.0, 2.0, 4.0
    is_playing: bool,
    interval_ms: u64,        // Base interval (e.g., 500ms)
}

impl StepPlayer {
    fn start_playback(&mut self) {
        self.is_playing = true;
        let interval = (self.interval_ms as f64 / self.speed) as u64;

        // Spawn async task to advance frames
        spawn(async move {
            while self.is_playing && self.current_index < self.steps.len() - 1 {
                gloo::timers::future::sleep(Duration::from_millis(interval)).await;
                self.current_index += 1;
            }
            self.is_playing = false;  // Auto-stop at end
        });
    }

    fn pause(&mut self) {
        self.is_playing = false;
    }

    fn step_to(&mut self, index: usize) {
        self.current_index = index.min(self.steps.len() - 1);
    }

    fn progress_percent(&self) -> f64 {
        (self.current_index as f64 / self.steps.len() as f64) * 100.0
    }
}
```

**B. Progress Indicators**

**Linear Progress Bar:**
```css
.progress-bar {
  width: 100%;
  height: 8px;
  background: #e0e0e0;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  transition: width 0.3s ease;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.8; }
}
```

**Step Counter with Context:**
```
Current State Indicators:

Simple:
  Step 127 of 280 (45%)

Detailed:
  Iteration 127 / 280
  Current Operation: Updating centroids
  Cost: 45.23 → 44.87 (↓ 0.8%)

Convergence:
  Iteration 127 / 280 (max)
  Convergence: 95% (threshold: 1e-4)
  [████████████████████░░] Nearly converged!
```

**C. Time Estimates**
```rust
struct TimeEstimator {
    iteration_durations: VecDeque<Duration>,  // Circular buffer (last 10 iters)
    max_history: usize,
}

impl TimeEstimator {
    fn new() -> Self {
        Self {
            iteration_durations: VecDeque::with_capacity(10),
            max_history: 10,
        }
    }

    fn record_iteration(&mut self, duration: Duration) {
        if self.iteration_durations.len() >= self.max_history {
            self.iteration_durations.pop_front();
        }
        self.iteration_durations.push_back(duration);
    }

    fn estimate_remaining(&self, current_step: usize, total_steps: usize) -> Duration {
        if self.iteration_durations.is_empty() {
            return Duration::from_secs(0);
        }

        // Average of last N iterations
        let avg_duration = self.iteration_durations.iter()
            .sum::<Duration>() / self.iteration_durations.len() as u32;

        let remaining_steps = total_steps - current_step;
        avg_duration * remaining_steps as u32
    }
}

// Display:
// "⏱ Estimated time remaining: ~3.2 seconds"
```

### 3.2 Snapshot Architecture

**A. Minimal State Snapshots**

**Problem:** Storing 1000 full algorithm states consumes too much memory

**Solution:** Differential snapshots
```rust
struct AlgorithmSnapshot {
    iteration: usize,
    changes: SnapshotChanges,  // Only what changed since last step
}

enum SnapshotChanges {
    KMeansUpdate {
        centroid_movements: Vec<(usize, Point, Point)>,  // (index, old, new)
        reassignments: Vec<(usize, usize, usize)>,       // (point_idx, old_cluster, new_cluster)
        inertia: f64,
    },

    GradientDescentUpdate {
        weight_delta: Vec<f64>,  // Just the change, not full weights
        cost: f64,
    },

    PCAUpdate {
        component_index: usize,
        variance_explained: f64,
    },
}

struct SnapshotPlayer {
    initial_state: AlgorithmState,  // Full state at start
    snapshots: Vec<AlgorithmSnapshot>,
    current_index: usize,
}

impl SnapshotPlayer {
    fn get_current_state(&self) -> AlgorithmState {
        // Reconstruct state by applying changes from start
        let mut state = self.initial_state.clone();

        for snapshot in &self.snapshots[..=self.current_index] {
            state.apply_changes(&snapshot.changes);
        }

        state
    }
}
```

**Memory Savings:** 10x reduction (1000 full states → 1 full + 999 diffs)

**B. Highlight Recent Changes**
```
Visual Pattern: Show What Just Happened

K-Means Visualization (Iteration 5 → 6):

Before (Step 5):
  Centroid A: (2.3, 4.1) [normal opacity]
  Centroid B: (5.7, 1.2) [normal opacity]

After (Step 6):
  Centroid A: (2.3, 4.1) [dim - no change]
  Centroid B: (5.9, 1.4) [bright + pulse animation] ← CHANGED

  Point #47: Cluster 0 → Cluster 1 [arrow animation]
  Point #53: Cluster 0 → Cluster 1 [arrow animation]
```

**CSS Animation:**
```css
.changed-element {
  animation: highlight-pulse 1s ease-out;
}

@keyframes highlight-pulse {
  0% {
    background-color: rgba(102, 126, 234, 0.3);
    transform: scale(1);
  }
  50% {
    background-color: rgba(102, 126, 234, 0.6);
    transform: scale(1.05);
  }
  100% {
    background-color: transparent;
    transform: scale(1);
  }
}
```

**C. Breakpoint System**
```
User Flow:
1. User clicks on iteration slider
2. Sets breakpoint at iteration 50
3. Clicks "Play"
4. Algorithm auto-pauses at iteration 50
5. User inspects state, then clicks "Continue"
```

**Implementation:**
```rust
struct BreakpointManager {
    breakpoints: HashSet<usize>,  // Iteration numbers
}

impl StepPlayer {
    fn advance_to_next_frame(&mut self, breakpoints: &BreakpointManager) {
        self.current_index += 1;

        // Check if we hit a breakpoint
        if breakpoints.has_breakpoint(self.current_index) {
            self.pause();
            console::log_1(&format!("⏸ Paused at breakpoint: iteration {}", self.current_index).into());
        }
    }
}
```

**UI Component:**
```rust
rsx! {
    div { class: "breakpoint-controls",
        h4 { "Breakpoints" }

        input {
            r#type: "number",
            placeholder: "Iteration number",
            min: "1",
            max: "{total_iterations}",
            value: "{new_breakpoint}",
            oninput: move |evt| new_breakpoint.set(evt.value())
        }

        button { onclick: move |_| add_breakpoint(), "Add Breakpoint" }

        ul {
            for bp in breakpoints.iter() {
                li {
                    "Iteration {bp}"
                    button { onclick: move |_| remove_breakpoint(bp), "✕" }
                }
            }
        }
    }
}
```

### 3.3 State Inspection Patterns

**A. Hover Tooltips**
```
Pattern: Rich Tooltips on Hover

K-Means Visualization:
  [User hovers over Centroid B]

  ┌─────────────────────────────┐
  │ Centroid B (Cluster 1)      │
  ├─────────────────────────────┤
  │ Position: (5.9, 1.4)        │
  │ Points in cluster: 47       │
  │ Inertia: 12.34              │
  │ Moved: 0.28 units this iter │
  └─────────────────────────────┘
```

**Implementation:**
```rust
use dioxus::prelude::*;

fn CentroidVisualization(cx: Scope, centroid: Centroid, index: usize) -> Element {
    let show_tooltip = use_state(cx, || false);

    cx.render(rsx! {
        div {
            class: "centroid",
            style: "left: {centroid.x * scale}px; top: {centroid.y * scale}px;",
            onmouseenter: move |_| show_tooltip.set(true),
            onmouseleave: move |_| show_tooltip.set(false),

            // Visual marker
            div { class: "centroid-marker" }

            // Tooltip (conditionally rendered)
            if **show_tooltip {
                rsx! {
                    div { class: "tooltip",
                        h4 { "Centroid {index} (Cluster {index})" }
                        p { "Position: ({centroid.x:.2}, {centroid.y:.2})" }
                        p { "Points in cluster: {centroid.num_points}" }
                        p { "Inertia: {centroid.inertia:.2}" }
                        p { "Moved: {centroid.movement:.2} units this iteration" }
                    }
                }
            }
        }
    })
}
```

**B. Side Panel State Inspector**
```
Layout: Visualization + Inspector

┌──────────────────┬──────────────────┐
│  Visualization   │  State Inspector │
│  (Canvas/SVG)    │                  │
│                  │  Iteration: 127  │
│    [Clusters]    │                  │
│    [Points]      │  Centroids:      │
│    [Centroids]   │  ├─ 0: (2.3,4.1)│
│                  │  ├─ 1: (5.9,1.4)│
│                  │  └─ 2: (8.1,7.3)│
│                  │                  │
│                  │  Inertia: 45.23  │
│                  │  Δ Inertia: -0.8%│
│                  │                  │
│                  │  Convergence:    │
│                  │  [████░░] 85%    │
└──────────────────┴──────────────────┘
```

**C. Diff View (Before/After)**
```
Pattern: Side-by-Side Comparison

Step 126 → Step 127: What Changed?

┌─────────────────────┬─────────────────────┐
│  Before (Step 126)  │  After (Step 127)   │
├─────────────────────┼─────────────────────┤
│ Centroid 0:         │ Centroid 0:         │
│   (2.30, 4.10)      │   (2.30, 4.10) ✓   │
│                     │   No change         │
│                     │                     │
│ Centroid 1:         │ Centroid 1:         │
│   (5.70, 1.20)      │   (5.90, 1.40) ⚠   │
│                     │   Moved 0.28 units  │
│                     │                     │
│ Points reassigned: 0│ Points reassigned: 2│
│                     │   #47: C0 → C1      │
│                     │   #53: C0 → C1      │
└─────────────────────┴─────────────────────┘
```

#### Best Practices Summary

1. **Video-Like Controls:** Play, pause, step forward/back, speed control (0.5x-4x)
2. **Progress Indicators:** Linear bar + step counter + time estimate
3. **Snapshot Architecture:** Store diffs, not full states (10x memory savings)
4. **Highlight Changes:** Animate/pulse elements that changed in last step
5. **Breakpoints:** Let users pause at specific iterations
6. **Hover Tooltips:** Rich information on hover
7. **Side Panel Inspector:** Persistent state display alongside visualization
8. **Diff View:** Show before/after comparisons for complex changes

---

## 4. Documentation Integration Patterns

### 4.1 Inline Help System

**Sources:** Chameleon, Userpilot, Nielsen Norman Group

#### Key Patterns

**A. Contextual Help Hierarchy**
```
Level 0: UI Labels (Always Visible)
  "Learning Rate:" [slider]

Level 1: Tooltips (On Hover)
  [?] → "Controls how fast the model learns. Higher = faster but less stable."

Level 2: Expandable Help (On Click)
  [Learn More ↓]
    → Expanded section with:
      - Detailed explanation
      - Recommended ranges (0.001 - 0.1)
      - Example: "For gradient descent, try 0.01"
      - Link to external resource

Level 3: External Documentation
  [📖 Read Full Guide] → Opens docs in new tab
```

**Implementation:**
```rust
fn LearningRateControl(cx: Scope) -> Element {
    let show_details = use_state(cx, || false);

    cx.render(rsx! {
        div { class: "parameter-control",
            // Level 0: Label
            label {
                "Learning Rate:"

                // Level 1: Tooltip
                span {
                    class: "tooltip-trigger",
                    title: "Controls how fast the model learns",
                    " [?]"
                }
            }

            // Control
            input { r#type: "range", min: "0.001", max: "0.1", step: "0.001" }

            // Level 2: Expandable details
            button {
                class: "help-toggle",
                onclick: move |_| show_details.set(!**show_details),
                if **show_details { "Hide Details ↑" } else { "Learn More ↓" }
            }

            if **show_details {
                rsx! {
                    div { class: "help-details",
                        p { "The learning rate determines the step size during gradient descent." }

                        div { class: "recommendation",
                            strong { "Recommended:" }
                            " Start with 0.01, then adjust based on convergence."
                        }

                        ul {
                            li { "Too high (> 0.1): Model may overshoot and diverge" }
                            li { "Too low (< 0.0001): Very slow convergence" }
                            li { "Just right (0.001-0.1): Steady improvement" }
                        }

                        // Level 3: External link
                        a {
                            href: "https://docs.example.com/learning-rate",
                            target: "_blank",
                            "📖 Read Full Guide"
                        }
                    }
                }
            }
        }
    })
}
```

**B. Tooltip Best Practices**
```
Golden Rules (Nielsen Norman Group):
1. Keep it brief (max 2 lines, ~100 characters)
2. Appear after 500ms hover (not instant)
3. Disappear on mouse leave
4. Don't block UI elements
5. Use consistent positioning (preferably above/below, not left/right)
```

**CSS Implementation:**
```css
.tooltip-trigger {
  position: relative;
  cursor: help;
  border-bottom: 1px dotted currentColor;
}

.tooltip-trigger:hover::after {
  content: attr(data-tooltip);
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  padding: 8px 12px;
  background: #333;
  color: white;
  border-radius: 4px;
  white-space: nowrap;
  font-size: 14px;
  z-index: 1000;
  animation: tooltip-fade-in 0.2s ease;
}

@keyframes tooltip-fade-in {
  from {
    opacity: 0;
    transform: translateX(-50%) translateY(-4px);
  }
  to {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }
}
```

**C. Contextual Banners**
```
Pattern: Announcement/Warning Banners

┌────────────────────────────────────────────────┐
│ ℹ️ New Feature: Try our PCA algorithm! [Try It] │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│ ⚠️ Large dataset detected. Processing may take │
│    30+ seconds. Consider sampling first.       │
│    [Dismiss] [Sample to 1000 rows]             │
└────────────────────────────────────────────────┘
```

**When to Use:**
- New feature announcements (info banner, dismissible)
- Warnings about performance (warning banner, actionable)
- Errors that need user attention (error banner, must be resolved)

**Implementation:**
```rust
#[derive(Clone, PartialEq)]
enum BannerType {
    Info,
    Warning,
    Error,
}

struct Banner {
    message: String,
    banner_type: BannerType,
    action: Option<(String, fn())>,  // Button text + callback
    dismissible: bool,
}

fn BannerComponent(cx: Scope, banner: Banner) -> Element {
    let is_visible = use_state(cx, || true);

    if !**is_visible {
        return None;
    }

    let icon = match banner.banner_type {
        BannerType::Info => "ℹ️",
        BannerType::Warning => "⚠️",
        BannerType::Error => "❌",
    };

    cx.render(rsx! {
        div {
            class: "banner banner-{banner.banner_type:?}",

            span { class: "banner-icon", "{icon}" }
            span { class: "banner-message", "{banner.message}" }

            if let Some((action_text, action_fn)) = &banner.action {
                button {
                    class: "banner-action",
                    onclick: move |_| action_fn(),
                    "{action_text}"
                }
            }

            if banner.dismissible {
                button {
                    class: "banner-dismiss",
                    onclick: move |_| is_visible.set(false),
                    "✕"
                }
            }
        }
    })
}
```

**D. Interactive Walkthroughs**
```
Pattern: First-Time User Onboarding

Step 1 of 4: Upload Data
┌─────────────────────────────────────┐
│                                     │
│   [Upload CSV]  ← Click here first │
│                                     │
│   [Skip Tour] [Next →]             │
└─────────────────────────────────────┘

Step 2 of 4: Select Algorithm
┌─────────────────────────────────────┐
│ [K-Means] [PCA] [LogReg]           │
│     ↑                               │
│  Choose an algorithm to run        │
│                                     │
│   [← Back] [Skip Tour] [Next →]   │
└─────────────────────────────────────┘
```

**Implementation:**
```rust
struct WalkthroughStep {
    title: String,
    message: String,
    target_element: String,  // CSS selector
    position: TooltipPosition,
}

enum TooltipPosition {
    Above,
    Below,
    Left,
    Right,
}

struct Walkthrough {
    steps: Vec<WalkthroughStep>,
    current_step: usize,
    is_active: bool,
}

fn WalkthroughOverlay(cx: Scope, walkthrough: &Walkthrough) -> Element {
    if !walkthrough.is_active {
        return None;
    }

    let step = &walkthrough.steps[walkthrough.current_step];

    cx.render(rsx! {
        // Dim overlay
        div { class: "walkthrough-overlay" }

        // Spotlight on target element
        div {
            class: "walkthrough-spotlight",
            style: "/* Position based on target_element */"
        }

        // Tooltip
        div {
            class: "walkthrough-tooltip",
            style: "/* Position relative to target */",

            h3 { "{step.title}" }
            p { "{step.message}" }

            div { class: "walkthrough-controls",
                button {
                    onclick: move |_| walkthrough.previous(),
                    disabled: walkthrough.current_step == 0,
                    "← Back"
                }

                button {
                    onclick: move |_| walkthrough.skip(),
                    "Skip Tour"
                }

                button {
                    onclick: move |_| walkthrough.next(),
                    if walkthrough.is_last_step() { "Finish" } else { "Next →" }
                }
            }

            div { class: "walkthrough-progress",
                "Step {walkthrough.current_step + 1} of {walkthrough.steps.len()}"
            }
        }
    })
}
```

**E. Embedded Help Centers**
```
Pattern: Collapsible Help Sidebar

┌────────────────┬──────────────────────────┐
│  Main Content  │  [?] Help               │
│                │                          │
│  [Visualization│  Common Questions:       │
│   here]        │  ▸ How does K-Means work?│
│                │  ▾ What's a good k value?│
│                │    Choose k=√(n/2) as    │
│                │    a starting point...   │
│                │  ▸ Why isn't it converging?│
│                │                          │
│                │  [Search help...]        │
└────────────────┴──────────────────────────┘
```

**Implementation:**
```rust
fn HelpSidebar(cx: Scope) -> Element {
    let is_open = use_state(cx, || false);
    let expanded_faqs = use_state(cx, || HashSet::new());

    cx.render(rsx! {
        div { class: "help-sidebar {is_open.then(|| \"open\")}",
            button {
                class: "help-toggle",
                onclick: move |_| is_open.set(!**is_open),
                "[?] Help"
            }

            if **is_open {
                rsx! {
                    div { class: "help-content",
                        h3 { "Common Questions" }

                        for (id, question, answer) in FAQ_ITEMS {
                            div { class: "faq-item",
                                button {
                                    class: "faq-question",
                                    onclick: move |_| toggle_faq(id),
                                    if expanded_faqs.contains(&id) { "▾" } else { "▸" }
                                    " {question}"
                                }

                                if expanded_faqs.contains(&id) {
                                    div { class: "faq-answer",
                                        dangerous_inner_html: answer  // Supports markdown
                                    }
                                }
                            }
                        }

                        input {
                            r#type: "text",
                            placeholder: "Search help...",
                            oninput: move |evt| search_help(evt.value())
                        }
                    }
                }
            }
        }
    })
}
```

### 4.2 Math Formula Rendering

**Sources:** KaTeX (Khan Academy), MathJax comparison

#### Key Insights

**A. KaTeX vs. MathJax**
```
Performance Comparison (Mobile):
- KaTeX: ~1 second to render page
- MathJax: ~30 seconds to render page

Feature Support:
- KaTeX: 90% of common LaTeX (fast, synchronous)
- MathJax: 100% LaTeX support (slow, asynchronous)

Recommendation: Use KaTeX for educational interfaces (speed matters)
```

**B. KaTeX Integration**
```html
<!-- Add KaTeX to your HTML -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>

<script>
  document.addEventListener("DOMContentLoaded", function() {
    renderMathInElement(document.body, {
      delimiters: [
        {left: "$$", right: "$$", display: true},
        {left: "$", right: "$", display: false}
      ]
    });
  });
</script>
```

**C. Common ML Formulas**
```latex
K-Means Cost Function:
$$J = \sum_{i=1}^{n} \sum_{j=1}^{k} w_{ij} ||x_i - \mu_j||^2$$

Gradient Descent Update:
$$\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)$$

Logistic Regression:
$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}}$$

PCA Variance Explained:
$$\text{Var}_k = \frac{\lambda_k}{\sum_{i=1}^{d} \lambda_i}$$
```

**D. Inline Formula Pattern**
```rust
// Rust+WASM: Use server-side rendering or inline SVG
fn render_formula(latex: &str) -> String {
    // Option 1: Pre-render formulas to SVG at build time
    // Option 2: Use katex-wasm (if available)
    // Option 3: Call JavaScript KaTeX from WASM

    format!(r#"<span class="katex-formula" data-latex="{}">{}</span>"#, latex, latex)
}

// In component:
rsx! {
    p {
        "The cost function is: "
        span { dangerous_inner_html: render_formula(r"J = \sum_{i=1}^{n} ||x_i - \mu_j||^2") }
    }
}
```

**E. Formula Tooltips**
```
Pattern: Hover for Explanation

Text: "We minimize the cost function $J$"
       [User hovers over J]

┌────────────────────────────────────┐
│ Cost Function (Inertia)            │
│                                    │
│ J = Σ ||xᵢ - μⱼ||²               │
│                                    │
│ Sum of squared distances from     │
│ each point to its assigned centroid│
└────────────────────────────────────┘
```

### 4.3 Code Snippet Display

**A. Syntax-Highlighted Code**
```rust
// Use prism.js or highlight.js for syntax highlighting
rsx! {
    div { class: "code-snippet",
        pre {
            code { class: "language-rust",
                r#"
let mut kmeans = KMeans::new(3, 100, 1e-4, Some(42));
kmeans.fit(&data)?;
let labels = kmeans.predict(&data)?;
                "#
            }
        }
    }
}
```

**B. Copy-to-Clipboard Button**
```rust
fn CodeSnippet(cx: Scope, code: String, language: String) -> Element {
    let copied = use_state(cx, || false);

    let copy_to_clipboard = move || {
        // Use clipboard API
        let _ = web_sys::window()
            .unwrap()
            .navigator()
            .clipboard()
            .write_text(&code);

        copied.set(true);

        // Reset after 2 seconds
        spawn(async move {
            gloo::timers::future::sleep(Duration::from_secs(2)).await;
            copied.set(false);
        });
    };

    cx.render(rsx! {
        div { class: "code-snippet",
            div { class: "code-header",
                span { class: "language-label", "{language}" }
                button {
                    class: "copy-button",
                    onclick: move |_| copy_to_clipboard(),
                    if **copied { "✓ Copied!" } else { "Copy" }
                }
            }

            pre {
                code { class: "language-{language}",
                    "{code}"
                }
            }
        }
    })
}
```

**C. Editable Code Playground**
```
Pattern: In-Browser Code Execution

┌─────────────────────────────────────┐
│ // Try changing k to see the effect │
│ let k = 3;                          │
│ let kmeans = KMeans::new(k, ...);   │
│                                     │
│ [▶ Run Code]                        │
├─────────────────────────────────────┤
│ Output:                             │
│ Clustered 150 points into 3 clusters│
│ Inertia: 78.85                      │
└─────────────────────────────────────┘
```

**Implementation Consideration:**
- Rust playgrounds require server-side execution (Rust Playground API)
- For WASM, could compile simplified examples to WASM and run locally
- Alternative: Show pseudo-code, not actual runnable Rust

#### Best Practices Summary

1. **Contextual Help Hierarchy:** Label → Tooltip → Expandable → External docs
2. **Tooltip Guidelines:** Brief (2 lines), 500ms delay, non-blocking position
3. **Banners:** Info/warning/error with icons, actionable buttons, dismissible
4. **Walkthroughs:** 4-step onboarding for first-time users, skippable
5. **Help Sidebar:** Collapsible FAQ + search, always accessible
6. **Math Rendering:** Use KaTeX (30x faster than MathJax)
7. **Code Snippets:** Syntax highlighting + copy button
8. **Editable Playgrounds:** For educational exploration (server-side for Rust)

---

## 5. WASM-Specific Considerations

### 5.1 WASM Panic Handling

**Sources:** rustwasm/console_error_panic_hook, Rust WASM forums, Stack Overflow

#### Critical Insights

**A. WASM Panics Are Fatal**
```
Problem:
1. Rust panic in WASM → Native WASM trap
2. Promise never resolves (leaked)
3. Entire WASM instance in unsafe state
4. No recovery possible (must reload)

Golden Rule: NEVER use .unwrap() in WASM hot paths
```

**B. Console Error Hook**
```rust
// Cargo.toml
[dependencies]
console_error_panic_hook = "0.1"

// main.rs or lib.rs
#[wasm_bindgen(start)]
pub fn main() {
    // Set panic hook for better debugging
    console_error_panic_hook::set_once();
}

// Now panics show in browser console instead of silent failure
```

**C. Error Boundary Pattern**
```rust
use std::panic;

// Wrap WASM entry points with panic catching
#[wasm_bindgen]
pub fn run_kmeans_safe(data_ptr: *const f64, n_samples: usize, n_features: usize, k: usize) -> String {
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        // Actual algorithm logic
        let data = unsafe { std::slice::from_raw_parts(data_ptr, n_samples * n_features) };
        let matrix = Matrix::from_slice(data, n_samples, n_features)?;

        let mut kmeans = KMeans::new(k, 100, 1e-4, Some(42));
        kmeans.fit(&matrix)?;

        Ok::<_, String>(format!("✅ K-Means completed! Clusters: {}", k))
    }));

    match result {
        Ok(Ok(success_msg)) => success_msg,
        Ok(Err(err_msg)) => format!("❌ Algorithm error: {}", err_msg),
        Err(_panic) => {
            web_sys::console::error_1(&"WASM panic caught!".into());
            "❌ Critical error. Please reload the page and try again.".to_string()
        }
    }
}
```

**D. UI-Level Error Boundaries**
```rust
// In Dioxus component
onclick: move |_| {
    spawn(async move {
        is_processing.set(true);

        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            run_algorithm(*selected_algorithm.read(), dataset)
        }));

        match result {
            Ok(msg) => {
                result_message.set(msg);
            }
            Err(_) => {
                result_message.set("❌ The algorithm crashed. Please reload and try simpler data.".to_string());
                web_sys::console::error_1(&"WASM panic in UI".into());

                // Optionally: Force page reload after 3 seconds
                spawn(async {
                    gloo::timers::future::sleep(Duration::from_secs(3)).await;
                    web_sys::window().unwrap().location().reload().ok();
                });
            }
        }

        is_processing.set(false);
    });
}
```

**E. Replace .unwrap() with Result Types**
```rust
// BAD: Will crash entire WASM instance
pub fn kmeans_fit(data: &Matrix<f64>) -> Vec<usize> {
    let centroids = initialize_centroids(data, k).unwrap();  // 💀 FATAL
    // ...
}

// GOOD: Propagates errors safely
pub fn kmeans_fit(data: &Matrix<f64>) -> Result<Vec<usize>, String> {
    let centroids = initialize_centroids(data, k)
        .map_err(|e| format!("Centroid initialization failed: {}", e))?;
    // ...
    Ok(labels)
}
```

### 5.2 Progress Indicators for Heavy Computation

**Sources:** Stack Overflow (WASM progress), Audjust blog, Web Workers patterns

#### Key Patterns

**A. Web Workers for Non-Blocking Computation**
```
Architecture:

Main Thread (UI):           Web Worker (WASM):
┌──────────────────┐       ┌──────────────────┐
│  User clicks     │──────→│  Load WASM       │
│  "Run K-Means"   │ msg   │  Run algorithm   │
│                  │       │                  │
│  Show progress:  │←──────│  Send progress:  │
│  "25% complete"  │ msg   │  every 10 iters  │
│                  │       │                  │
│  Display result  │←──────│  Send final      │
│                  │ msg   │  result          │
└──────────────────┘       └──────────────────┘
```

**Implementation:**
```javascript
// worker.js
importScripts('wasm_module.js');

self.onmessage = async (event) => {
  const { data, k, max_iterations } = event.data;

  // Load WASM module
  const wasm = await wasm_bindgen('./wasm_module_bg.wasm');

  // Run algorithm with progress callback
  wasm.run_kmeans_with_progress(
    data,
    k,
    max_iterations,
    (iteration, total) => {
      // Send progress update to main thread
      self.postMessage({
        type: 'progress',
        iteration,
        total,
        percent: (iteration / total) * 100
      });
    }
  );

  // Send final result
  self.postMessage({
    type: 'complete',
    labels: result.labels,
    centroids: result.centroids
  });
};
```

```rust
// In WASM module
#[wasm_bindgen]
pub fn run_kmeans_with_progress(
    data: Vec<f64>,
    k: usize,
    max_iterations: usize,
    progress_callback: &js_sys::Function,
) -> Result<JsValue, JsValue> {
    let matrix = Matrix::from_vec(data, n_samples, n_features)?;
    let mut kmeans = KMeans::new(k, max_iterations, 1e-4, None);

    for iteration in 0..max_iterations {
        // ... K-Means iteration logic ...

        // Every 10 iterations, call progress callback
        if iteration % 10 == 0 {
            let this = &JsValue::NULL;
            let _ = progress_callback.call2(
                this,
                &JsValue::from(iteration),
                &JsValue::from(max_iterations)
            );
        }

        if converged {
            break;
        }
    }

    // Return result as JsValue
    Ok(serde_wasm_bindgen::to_value(&kmeans)?)
}
```

**B. Main Thread: Chunked Execution**
```rust
// For algorithms that can be chunked (not blocking)
async fn run_kmeans_chunked(data: Matrix<f64>, k: usize) {
    let mut kmeans = KMeans::new(k, 1000, 1e-4, None);

    for chunk in 0..100 {  // 100 chunks of 10 iterations each
        // Run 10 iterations
        for _ in 0..10 {
            kmeans.step_once()?;  // Custom method: single iteration
        }

        // Update progress
        let progress = ((chunk + 1) * 10) as f64 / 1000.0;
        progress_signal.set(progress);

        // Yield to browser (allow UI updates)
        gloo::timers::future::sleep(Duration::from_millis(0)).await;
    }
}
```

**C. Progress UI Components**
```rust
fn ProgressIndicator(cx: Scope) -> Element {
    let progress = use_context::<Signal<f64>>(cx);  // 0.0 to 1.0
    let is_processing = use_context::<Signal<bool>>(cx);

    if !**is_processing.read() {
        return None;
    }

    cx.render(rsx! {
        div { class: "progress-overlay",
            div { class: "progress-card",
                h3 { "Processing..." }

                // Progress bar
                div { class: "progress-bar",
                    div {
                        class: "progress-fill",
                        style: "width: {progress * 100.0}%"
                    }
                }

                // Percentage
                p { class: "progress-text",
                    "{(progress * 100.0).round()}% complete"
                }

                // Spinner animation
                div { class: "spinner" }

                // Cancel button
                button {
                    onclick: move |_| cancel_processing(),
                    "Cancel"
                }
            }
        }
    })
}
```

**D. Timeout Protection**
```rust
use std::time::Instant;

pub fn run_algorithm_with_timeout(
    algorithm: Algorithm,
    data: &Matrix<f64>,
    timeout_secs: u64,
) -> Result<String, String> {
    let start = Instant::now();
    let timeout = Duration::from_secs(timeout_secs);

    match algorithm {
        Algorithm::KMeans => {
            let mut kmeans = KMeans::new(3, 1000, 1e-4, None);

            for iteration in 0..1000 {
                if start.elapsed() > timeout {
                    return Err(format!(
                        "⏱ Algorithm timed out after {} seconds (iteration {}). Try reducing dataset size.",
                        timeout_secs, iteration
                    ));
                }

                kmeans.step_once()?;

                if kmeans.has_converged() {
                    break;
                }
            }

            Ok(format!("✅ K-Means converged in {} iterations", kmeans.n_iterations))
        }
        // ... other algorithms
    }
}
```

### 5.3 Memory Constraints

**A. Input Size Limits**
```rust
const MAX_CSV_SIZE: usize = 5 * 1024 * 1024;  // 5 MB
const MAX_ROWS: usize = 10_000;
const MAX_FEATURES: usize = 100;

pub fn validate_csv_upload(file_size: usize, rows: usize, features: usize) -> Result<(), String> {
    if file_size > MAX_CSV_SIZE {
        return Err(format!(
            "❌ File too large: {:.1}MB. Maximum: 5MB",
            file_size as f64 / 1_024_000.0
        ));
    }

    if rows > MAX_ROWS {
        return Err(format!(
            "❌ Too many rows: {}. Maximum: {}. Consider sampling.",
            rows, MAX_ROWS
        ));
    }

    if features > MAX_FEATURES {
        return Err(format!(
            "❌ Too many features: {}. Maximum: {}",
            features, MAX_FEATURES
        ));
    }

    Ok(())
}
```

**B. Circular Buffers for History**
```rust
use std::collections::VecDeque;

struct LossHistory {
    values: VecDeque<f64>,
    max_size: usize,
}

impl LossHistory {
    fn new(max_size: usize) -> Self {
        Self {
            values: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    fn push(&mut self, value: f64) {
        if self.values.len() >= self.max_size {
            self.values.pop_front();  // Remove oldest
        }
        self.values.push_back(value);
    }

    fn get_recent(&self, n: usize) -> Vec<f64> {
        self.values.iter()
            .rev()
            .take(n)
            .copied()
            .collect()
    }
}

// Usage:
const MAX_LOSS_HISTORY: usize = 10_000;
let mut loss_history = LossHistory::new(MAX_LOSS_HISTORY);

for iteration in 0..1_000_000 {
    let loss = compute_loss();
    loss_history.push(loss);  // Automatically caps at 10K entries
}
```

**C. Memory Profiling UI**
```rust
#[wasm_bindgen]
pub fn get_memory_usage() -> f64 {
    let performance = web_sys::window()
        .unwrap()
        .performance()
        .unwrap();

    let memory = performance
        .memory()
        .unwrap();

    memory.used_js_heap_size() as f64 / 1_048_576.0  // Convert to MB
}

// In UI:
fn MemoryMonitor(cx: Scope) -> Element {
    let memory_mb = use_signal(cx, || 0.0);

    use_future(cx, (), |_| {
        to_owned![memory_mb];
        async move {
            loop {
                gloo::timers::future::sleep(Duration::from_secs(1)).await;
                memory_mb.set(get_memory_usage());
            }
        }
    });

    cx.render(rsx! {
        div { class: "memory-monitor",
            "Memory: {memory_mb:.1} MB"

            if *memory_mb.read() > 100.0 {
                span { class: "warning", " ⚠️ High memory usage" }
            }
        }
    })
}
```

#### Best Practices Summary

1. **Panic Handling:** Use console_error_panic_hook, wrap entry points with catch_unwind
2. **Error Boundaries:** Catch panics in UI, show user-friendly messages, offer reload
3. **Result Types:** NEVER use .unwrap() in WASM, always return Result<T, String>
4. **Web Workers:** Offload heavy computation to workers, send progress updates
5. **Chunked Execution:** For main thread, yield every N iterations with sleep(0ms)
6. **Progress UI:** Show percentage, spinner, estimated time, cancel button
7. **Timeouts:** Limit algorithm execution to 5-10 seconds max
8. **Input Limits:** Enforce CSV size (5MB), rows (10K), features (100) limits
9. **Circular Buffers:** Cap history to prevent unbounded memory growth
10. **Memory Monitoring:** Show memory usage in dev mode, warn at high levels

---

## 6. Implementation Roadmap for ML Playground

### Phase 1: Foundation (Week 1)

**Priority 1: WASM Safety**
- [ ] Add console_error_panic_hook to all WASM entry points
- [ ] Replace all .unwrap() calls with Result types
- [ ] Implement error boundaries in UI components
- [ ] Add input validation (CSV size, row/column limits)
- [ ] Add algorithm timeouts (5 second max)

**Priority 2: Data Exploration UI**
- [ ] CSV preview table with 25-row pagination
- [ ] Automatic column type inference
- [ ] Type badges (Integer, Float, Boolean, Categorical)
- [ ] Multi-column sorting
- [ ] Real-time search filter

**Priority 3: Feature Selection**
- [ ] Checkbox-based feature selector
- [ ] Minimum 2 features validation
- [ ] "Select All" / "Clear All" buttons
- [ ] Show selected count in UI

### Phase 2: Educational Features (Week 2-3)

**Priority 1: Algorithm Step-Through**
- [ ] Snapshot-based architecture for K-Means
- [ ] Video-like controls (play, pause, step forward/back)
- [ ] Speed control (0.5x, 1x, 2x, 4x)
- [ ] Progress bar with step counter
- [ ] Highlight changes between steps

**Priority 2: Inline Help System**
- [ ] Tooltips for all parameters (learning rate, k clusters, etc.)
- [ ] Expandable "Learn More" sections
- [ ] Contextual banners for warnings (large dataset, etc.)
- [ ] Help sidebar with FAQ

**Priority 3: Math Formula Rendering**
- [ ] Integrate KaTeX for LaTeX rendering
- [ ] Add formulas to algorithm descriptions (K-Means cost function, etc.)
- [ ] Hover tooltips for formula explanations

### Phase 3: Advanced Interactions (Week 4-5)

**Priority 1: Interactive Hyperparameters**
- [ ] Sliders for k clusters (2-10 range)
- [ ] Learning rate slider (logarithmic scale)
- [ ] Max iterations slider
- [ ] Real-time preview of parameter effects

**Priority 2: Progress Indicators**
- [ ] Web Worker for K-Means (if > 1000 samples)
- [ ] Progress overlay with percentage
- [ ] Time estimation (elapsed + remaining)
- [ ] Cancel button

**Priority 3: Comparative Visualization**
- [ ] Side-by-side algorithm comparison (StandardScaler vs MinMaxScaler)
- [ ] Timing benchmarks displayed in UI
- [ ] "Better for:" use case recommendations

### Phase 4: Polish & Delight (Week 6-8)

**Priority 1: Onboarding**
- [ ] 4-step walkthrough for first-time users
- [ ] Interactive tutorial with sample dataset
- [ ] Dismissible tips (localStorage to remember)

**Priority 2: Export & Sharing**
- [ ] Export results as JSON
- [ ] Copy model coefficients to clipboard
- [ ] Export visualizations as PNG
- [ ] Shareable URLs with configurations

**Priority 3: Performance**
- [ ] Zero-allocation hot paths (see CLAUDE.md recommendations)
- [ ] Canvas migration for > 1000 points
- [ ] Virtual scrolling for large tables
- [ ] Memory monitoring in dev mode

**Priority 4: Accessibility**
- [ ] ARIA labels for all controls
- [ ] Keyboard navigation (Tab, Arrow keys, Space, Enter)
- [ ] Screen reader announcements for algorithm progress
- [ ] High-contrast mode

---

## 7. Concrete Examples from Successful Platforms

### Example 1: TensorFlow Playground - Dataset Selection

**Visual Pattern:**
```
Data Section (Left Panel):
┌────────────────────────────┐
│ DATA                       │
├────────────────────────────┤
│ Dataset:                   │
│  ( ) Circle                │
│  (•) XOR                   │
│  ( ) Gaussian              │
│  ( ) Spiral                │
│                            │
│ Noise:  [====    ] 25%     │
│ Ratio:  [======  ] 50%     │
└────────────────────────────┘
```

**Our Equivalent:**
```rust
rsx! {
    div { class: "data-section",
        h3 { "Data Source" }

        // Dataset selection
        label {
            input { r#type: "radio", name: "dataset", value: "iris", checked: true }
            "Iris Dataset (150 samples, 4 features)"
        }
        label {
            input { r#type: "radio", name: "dataset", value: "wine" }
            "Wine Dataset (178 samples, 13 features)"
        }
        label {
            input { r#type: "radio", name: "dataset", value: "custom" }
            "Upload CSV"
        }

        // Noise slider (if synthetic data)
        if dataset == "synthetic" {
            div { class: "slider-control",
                label { "Noise Level:" }
                input { r#type: "range", min: "0", max: "100", value: "25" }
                span { "25%" }
            }
        }
    }
}
```

---

### Example 2: Distill.pub - Interactive Diagram

**Article:** Feature Visualization (distill.pub/2017/feature-visualization/)

**Pattern: Slider-Controlled Image Grid**
```
[Slider: Diversity Term] ──────────── 0.0 to 1.0

Without Diversity (0.0):        With Diversity (1.0):
┌──────────────────┐            ┌──────────────────┐
│ [Single blurry   │            │ [Four distinct   │
│  visualization]  │            │  visualizations] │
└──────────────────┘            └──────────────────┘

Caption: "Adding a diversity term reveals multiple facets of what the neuron detects."
```

**Our Equivalent for K-Means:**
```rust
rsx! {
    div { class: "interactive-comparison",
        h3 { "Effect of Number of Clusters (k)" }

        input {
            r#type: "range",
            min: "2",
            max: "10",
            value: "{k}",
            oninput: move |evt| {
                k.set(evt.value().parse().unwrap_or(3));
                rerun_kmeans();  // Trigger re-computation
            }
        }
        span { "k = {k}" }

        div { class: "visualization-grid",
            // Left: k=3 (reference)
            div { class: "viz-panel",
                h4 { "k = 3 (baseline)" }
                ScatterPlot { data: data.clone(), labels: labels_k3, k: 3 }
                p { "Inertia: {inertia_k3:.2}" }
            }

            // Right: Current k value
            div { class: "viz-panel",
                h4 { "k = {k} (current)" }
                ScatterPlot { data: data.clone(), labels: labels_current, k: *k }
                p { "Inertia: {inertia_current:.2}" }
            }
        }

        p { class: "insight",
            if *k < 3 {
                "⚠️ Too few clusters - high inertia (poor fit)"
            } else if *k > 6 {
                "⚠️ Too many clusters - overfitting likely"
            } else {
                "✓ Good cluster count for this dataset"
            }
        }
    }
}
```

---

### Example 3: Observable - Reactive Inputs

**Pattern: Checkbox + Immediate Update**
```javascript
viewof selected_features = Inputs.checkbox(
  ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  {label: "Features:", value: ["sepal_length", "petal_length"]}
)

// This cell automatically updates when selected_features changes
filtered_data = data.map(d =>
  Object.fromEntries(
    selected_features.map(f => [f, d[f]])
  )
)

Plot.dot(filtered_data, {x: selected_features[0], y: selected_features[1]})
```

**Our Dioxus Equivalent:**
```rust
let mut selected_features = use_signal(|| vec!["sepal_length".to_string(), "petal_length".to_string()]);

rsx! {
    div { class: "feature-selector",
        h4 { "Features:" }

        for feature in ALL_FEATURES {
            label {
                input {
                    r#type: "checkbox",
                    checked: selected_features.read().contains(&feature.to_string()),
                    onchange: move |evt| {
                        let mut features = selected_features.write();
                        if evt.checked() {
                            features.push(feature.to_string());
                        } else {
                            features.retain(|f| f != feature);
                        }
                    }
                }
                span { "{feature}" }
            }
        }
    }

    // Automatically updates when selected_features changes
    ScatterPlot {
        data: data.clone(),
        x_feature: selected_features.read().get(0).cloned(),
        y_feature: selected_features.read().get(1).cloned(),
    }
}
```

---

### Example 4: VisuAlgo - Algorithm State Inspector

**Pattern: Side Panel with Tree View**
```
┌──────────────────┬──────────────────┐
│  Visualization   │  Algorithm State │
├──────────────────┼──────────────────┤
│                  │ Iteration: 5     │
│   [Sorting viz]  │                  │
│                  │ Array:           │
│                  │ ▾ Sorted: [1,2]  │
│                  │ ▸ Unsorted:      │
│                  │     [5,3,8,4]    │
│                  │                  │
│                  │ Comparisons: 12  │
│                  │ Swaps: 7         │
└──────────────────┴──────────────────┘
```

**Our K-Means Equivalent:**
```rust
rsx! {
    div { class: "app-layout",
        // Left: Visualization
        div { class: "visualization-panel",
            ScatterPlot { data: data, labels: current_labels, centroids: current_centroids }
        }

        // Right: State Inspector
        div { class: "state-inspector",
            h3 { "Algorithm State" }

            p { "Iteration: {iteration} / {max_iterations}" }

            details { open: true,
                summary { "Centroids ({k})" }
                ul {
                    for (i, centroid) in current_centroids.iter().enumerate() {
                        li {
                            "Cluster {i}: ({centroid.x:.2}, {centroid.y:.2})"
                            span { class: "point-count", " ({centroid.num_points} points)" }
                        }
                    }
                }
            }

            details {
                summary { "Convergence" }
                p { "Inertia: {inertia:.2}" }
                p { "Δ Inertia: {delta_inertia:.2}%" }

                div { class: "convergence-bar",
                    div {
                        class: "fill",
                        style: "width: {convergence_percent}%"
                    }
                }
            }

            details {
                summary { "Statistics" }
                p { "Cluster reassignments: {reassignments}" }
                p { "Total distance moved: {total_movement:.2}" }
            }
        }
    }
}
```

---

### Example 5: Python Tutor - Code + Memory Split Panel

**Pattern: Synchronized Highlighting**
```
Code Panel:                   Memory Panel:
┌─────────────────────┐      ┌─────────────────────┐
│ 1  def kmeans(...): │      │ Global frame:       │
│ 2    centroids = [] │      │                     │
│ 3►   for i in ...:  │ ◄────┤ kmeans frame:       │
│ 4      centroids... │      │   i: 2 ◄────────────┤
└─────────────────────┘      │   centroids: [...]  │
                              └─────────────────────┘
```

**Our Pseudo-Code Equivalent:**
```rust
rsx! {
    div { class: "split-panel",
        // Left: Algorithm Pseudo-Code
        div { class: "code-panel",
            h3 { "K-Means Algorithm" }
            pre {
                code {
                    "1. Initialize k random centroids\n"
                    "2. REPEAT until convergence:\n"
                    if current_step == "assignment" {
                        "3.►  Assign each point to nearest centroid\n"
                    } else {
                        "3.   Assign each point to nearest centroid\n"
                    }

                    if current_step == "update" {
                        "4.►  Update centroids to cluster means\n"
                    } else {
                        "4.   Update centroids to cluster means\n"
                    }

                    "5.   Check if converged\n"
                }
            }
        }

        // Right: Current State
        div { class: "state-panel",
            h3 { "Current State" }

            if current_step == "assignment" {
                div {
                    p { "Assigning point #{current_point}:" }
                    p { "  Point: ({point.x:.2}, {point.y:.2})" }
                    p { "  Nearest centroid: {nearest_centroid}" }
                    p { "  Distance: {distance:.2}" }
                }
            } else if current_step == "update" {
                div {
                    p { "Updating centroid #{current_centroid}:" }
                    p { "  Old position: ({old_pos.x:.2}, {old_pos.y:.2})" }
                    p { "  New position: ({new_pos.x:.2}, {new_pos.y:.2})" }
                    p { "  Movement: {movement:.2} units" }
                }
            }
        }
    }
}
```

---

## 8. Key Takeaways & Quick Reference

### The Golden Rules

1. **Immediate Feedback** - Updates in < 100ms feel instant
2. **Orange/Blue Standard** - Negative = orange, positive = blue
3. **Progressive Disclosure** - Show simple first, reveal complexity on demand
4. **Video Controls** - Play/pause/step for algorithm execution
5. **Result Over Panic** - WASM: Use Result<T, E>, never .unwrap()
6. **Circular Buffers** - Prevent unbounded memory growth
7. **25-Row Tables** - Optimal default pagination
8. **Tooltips at 500ms** - Not instant, not too slow
9. **KaTeX for Math** - 30x faster than MathJax
10. **Web Workers for Heavy Lifting** - Keep main thread responsive

### UI Component Checklist

For **every** ML algorithm visualization:
- [ ] Play/pause/step controls
- [ ] Progress indicator (percentage + time estimate)
- [ ] Parameter sliders with tooltips
- [ ] State inspector panel
- [ ] Highlight recent changes (pulse animation)
- [ ] Error boundaries with user-friendly messages
- [ ] Input validation and size limits
- [ ] Help tooltip on every control
- [ ] Keyboard shortcuts (Space, Arrow keys)
- [ ] Mobile-responsive layout (cards on small screens)

### Common Pitfalls to Avoid

1. ❌ Using .unwrap() in WASM → Use Result types
2. ❌ Unbounded history arrays → Use circular buffers
3. ❌ Blocking main thread → Use Web Workers or chunked execution
4. ❌ Silent errors → Always show user-friendly error messages
5. ❌ No progress indicators → Always show progress for > 1 second operations
6. ❌ Complex UI upfront → Use progressive disclosure
7. ❌ No input validation → Enforce CSV size and dimension limits
8. ❌ Mathematical jargon → Explain in plain language, math in tooltips
9. ❌ Fixed layouts → Design mobile-first, enhance for desktop
10. ❌ No keyboard support → Accessibility is not optional

### Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Initial load | < 3s | Time to interactive |
| Algorithm start | < 100ms | Click to first frame |
| Frame rate | 60 FPS | During animation |
| Iteration rate | 1000+/sec | For lightweight algorithms |
| Memory | < 100 MB | WASM heap size |
| CSV upload | < 5 MB | File size limit |
| Max rows | 10,000 | Prevent browser freeze |
| Timeout | 5 seconds | Kill runaway algorithms |

---

## 9. Recommended Libraries & Tools

### Rust/WASM

| Tool | Purpose | Why |
|------|---------|-----|
| `console_error_panic_hook` | Better panic debugging | Shows panics in browser console |
| `wasm-bindgen` | JS ↔ Rust interop | Industry standard |
| `gloo-timers` | Async timers | `sleep(0ms)` to yield to browser |
| `web-sys` | Browser API access | Clipboard, console, window, etc. |
| `serde-wasm-bindgen` | Serialize to JS | Pass complex types to JS |

### JavaScript (for complementary features)

| Tool | Purpose | Why |
|------|---------|-----|
| KaTeX | Math rendering | 30x faster than MathJax |
| Prism.js | Code syntax highlighting | Lightweight, extensible |
| Web Workers API | Background computation | Keep UI responsive |

### CSS Frameworks (Optional)

| Tool | Purpose | Why |
|------|---------|-----|
| Tailwind CSS | Utility-first styling | Rapid prototyping |
| Custom CSS | Full control | Best for educational UIs (see main.css) |

### Testing

| Tool | Purpose | Why |
|------|---------|-----|
| Playwright | E2E testing | Browser automation |
| wasm-bindgen-test | WASM unit tests | Test WASM in headless browser |

---

## 10. Further Reading

### Official Documentation
- [TensorFlow Playground](https://playground.tensorflow.org/) - Live demo
- [Distill.pub](https://distill.pub/) - Archive of interactive ML articles
- [Observable](https://observablehq.com/) - Reactive notebooks
- [VisuAlgo](https://visualgo.net/) - Algorithm visualizations
- [Python Tutor](https://pythontutor.com/) - Code execution visualizer

### Design Guidelines
- [Nielsen Norman Group - Progressive Disclosure](https://www.nngroup.com/articles/progressive-disclosure/)
- [Tooltips UX Best Practices](https://www.appcues.com/blog/tooltips)
- [Data Table Design Patterns](https://www.pencilandpaper.io/articles/ux-pattern-analysis-enterprise-data-tables)

### WASM Best Practices
- [rustwasm Book](https://rustwasm.github.io/book/)
- [console_error_panic_hook](https://github.com/rustwasm/console_error_panic_hook)
- [WASM Performance Patterns](https://web.dev/articles/webassembly-performance-patterns-for-web-apps)

### Math Rendering
- [KaTeX Documentation](https://katex.org/)
- [KaTeX vs MathJax Performance](https://www.intmath.com/blog/mathematics/katex-a-new-way-to-display-math-on-the-web-9445)

### Academic Research
- [VisuAlgo Paper (ResearchGate)](https://www.researchgate.net/publication/282210883_VisuAlgo_-_Visualising_Data_Structures_and_Algorithms_Through_Animation)
- [Semi-automatic Column Type Inference for CSV](https://link.springer.com/chapter/10.1007/978-3-030-67731-2_39)

---

## Appendix A: Color Palette Recommendations

### Educational ML Interfaces

**Primary Colors (Based on TensorFlow Playground):**
```css
:root {
  /* Data/Predictions */
  --color-positive: #3B82F6;      /* Blue - positive values/class 1 */
  --color-negative: #F59E0B;      /* Orange - negative values/class 0 */

  /* UI Accents */
  --color-primary: #667EEA;       /* Purple - primary actions */
  --color-secondary: #764BA2;     /* Deep purple - secondary actions */

  /* Neutral */
  --color-background: #FFFFFF;
  --color-text: #1F2937;
  --color-border: #E5E7EB;

  /* Semantic */
  --color-success: #10B981;       /* Green - success states */
  --color-warning: #F59E0B;       /* Amber - warnings */
  --color-error: #EF4444;         /* Red - errors */
  --color-info: #3B82F6;          /* Blue - information */
}
```

**Gradient for Progress Bars:**
```css
.progress-fill {
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}
```

**Diverging Scale (Correlations: -1 to +1):**
```css
/* Red → White → Blue */
--color-correlation-negative: #EF4444;  /* Strong negative (-1) */
--color-correlation-neutral: #FFFFFF;   /* No correlation (0) */
--color-correlation-positive: #3B82F6;  /* Strong positive (+1) */
```

---

## Appendix B: Accessibility Checklist

### WCAG 2.1 AA Compliance

- [ ] **Color Contrast:** 4.5:1 for normal text, 3:1 for large text
- [ ] **Keyboard Navigation:** All controls accessible via Tab, Arrow keys
- [ ] **Focus Indicators:** Visible focus outline (not `outline: none`)
- [ ] **ARIA Labels:** `aria-label` or `aria-labelledby` on all interactive elements
- [ ] **Screen Reader Announcements:** Use `aria-live` for dynamic updates
- [ ] **Skip Links:** "Skip to main content" link at top
- [ ] **Heading Hierarchy:** Proper h1 → h2 → h3 structure
- [ ] **Alt Text:** Descriptive alt text for all images/visualizations
- [ ] **Form Labels:** Every input has associated `<label>`
- [ ] **Error Messages:** Clear, associated with invalid fields

### Keyboard Shortcuts (Suggested)

| Key | Action |
|-----|--------|
| Space | Play/Pause algorithm |
| Left Arrow | Previous step |
| Right Arrow | Next step |
| Home | Jump to first step |
| End | Jump to last step |
| ? | Show help overlay |
| Esc | Close modals/overlays |

---

**End of Research Document**

**Document Stats:**
- **Word Count:** ~15,000 words
- **Code Examples:** 50+
- **Platforms Analyzed:** 5 (TensorFlow Playground, Distill.pub, Observable, VisuAlgo, Python Tutor)
- **Best Practices Identified:** 100+
- **Implementation Examples:** 30+

**Next Steps:**
1. Review with team
2. Prioritize patterns for ML Playground v0.2
3. Create component library based on patterns
4. Implement Phase 1 (Foundation) from roadmap
