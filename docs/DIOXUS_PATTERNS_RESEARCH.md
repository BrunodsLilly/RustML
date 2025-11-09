# Dioxus Framework Documentation Research

**Research Focus:** Building interactive result displays for ML Playground
**Date:** November 9, 2025
**Target:** Complex component composition, state management, conditional rendering, tables, charts, tabs

---

## Table of Contents

1. [Signal-Based State Management](#1-signal-based-state-management)
2. [Complex Component Composition](#2-complex-component-composition)
3. [Conditional Rendering Patterns](#3-conditional-rendering-patterns)
4. [Table and Data Grid Patterns](#4-table-and-data-grid-patterns)
5. [Chart and Visualization Patterns](#5-chart-and-visualization-patterns)
6. [Tab and Accordion Patterns](#6-tab-and-accordion-patterns)
7. [Performance Optimization](#7-performance-optimization)
8. [Styling and CSS Integration](#8-styling-and-css-integration)
9. [Recommended Patterns for ML Playground](#9-recommended-patterns-for-ml-playground)

---

## 1. Signal-Based State Management

### 1.1 Core Concepts

Dioxus 0.6+ uses **signals** as the primary state management mechanism. Signals are reactive, `Copy`, and `'static`, eliminating lifetime issues.

#### Basic Signal Usage

```rust
use dioxus::prelude::*;

#[component]
fn Counter() -> Element {
    // use_signal creates reactive state
    let mut count = use_signal(|| 0);

    rsx! {
        div {
            h1 { "Count: {count}" }
            button {
                onclick: move |_| count += 1,
                "Increment"
            }
        }
    }
}
```

**Key Features:**
- **Automatic tracking:** Components re-render when reading signal values
- **Copy semantics:** Signals can be moved into closures without `.clone()`
- **Helper methods:** Call signal as function to clone value: `let val = count()`

### 1.2 Advanced Signal Patterns

#### Multiple Independent Signals

```rust
#[component]
fn MultiState() -> Element {
    let mut count = use_signal(|| 0);
    let mut toggled = use_signal(|| false);

    rsx! {
        h2 { "Count: {count}" }
        button { onclick: move |_| *count.write() += 1, "Increment" }

        h2 { "Toggled: {toggled}" }
        button { onclick: move |_| *toggled.write() = !*toggled.read(), "Toggle" }
    }
}
```

#### Complex State (Structs and Collections)

```rust
#[derive(Clone, Debug)]
struct EditorState {
    text: String,
    cursor: usize,
}

impl EditorState {
    fn handle_input(&mut self, event: FormEvent) {
        self.text = event.value();
    }
}

#[component]
fn TextEditor() -> Element {
    let mut state = use_signal(|| EditorState {
        text: String::new(),
        cursor: 0,
    });

    rsx! {
        input {
            value: "{state.read().text}",
            oninput: move |evt| state.write().handle_input(evt)
        }
    }
}
```

#### Vector/List State

```rust
#[component]
fn ItemList() -> Element {
    let mut items = use_signal(|| vec!["Apple", "Banana", "Cherry"]);
    let mut input = use_signal(|| String::new());

    rsx! {
        input {
            value: "{input}",
            oninput: move |evt| input.set(evt.value()),
        }
        button {
            onclick: move |_| {
                if !input().is_empty() {
                    items.push(input());
                    input.set(String::new());
                }
            },
            "Add Item"
        }
        ul {
            for (idx, item) in items.iter().enumerate() {
                li {
                    key: "{idx}",
                    "{item}"
                    button {
                        onclick: move |_| { items.remove(idx); },
                        "Remove"
                    }
                }
            }
        }
    }
}
```

**IMPORTANT:** Always use `key` attribute for list items to ensure correct rendering.

### 1.3 Global State

#### Global Signals

```rust
use dioxus::prelude::*;

// Define global state at module level
pub static DARK_MODE: GlobalSignal<Option<bool>> = Signal::global(|| None);
pub static SHOW_SEARCH: GlobalSignal<bool> = Signal::global(|| false);

// Access in any component without props
#[component]
fn DarkModeToggle() -> Element {
    let mut dark_mode = DARK_MODE;

    use_effect(move || {
        if dark_mode() {
            document::eval(r#"document.documentElement.classList.add('dark')"#);
        } else {
            document::eval(r#"document.documentElement.classList.remove('dark')"#);
        }
    });

    rsx! {
        button {
            onclick: move |_| dark_mode.set(Some(!dark_mode())),
            if dark_mode() { "🌙 Dark" } else { "☀️ Light" }
        }
    }
}
```

**Benefits:**
- No need for Context API for simple global state
- Simpler than external state management libraries
- Automatic reactivity across the entire app

#### Context API with Signals

For shared state between component trees:

```rust
fn Parent() -> Element {
    // Provide signal to context
    let mut state = use_context_provider(|| Signal::new(0));

    rsx! {
        button { onclick: move |_| state += 1, "Increment" }
        Child {}
    }
}

fn Child() -> Element {
    // Consume from context
    let state = use_context::<Signal<i32>>();
    rsx! { "{state}" }
}
```

### 1.4 Derived State with `use_memo`

**Use Case:** Expensive computations that depend on reactive values

```rust
#[component]
fn ExpensiveComputation() -> Element {
    let mut count = use_signal(|| 0);

    // Only recomputes when count changes
    let doubled = use_memo(move || count() * 2);
    let squared = use_memo(move || count() * count());

    rsx! {
        div {
            "Count: {count}"
            "Doubled: {doubled}"
            "Squared: {squared}"
            button { onclick: move |_| count += 1, "Increment" }
        }
    }
}
```

**Key Insight:** Memos only trigger downstream updates if their output value changes.

```rust
let mut count = use_signal(|| 1);
let double_count = use_memo(move || count() * 2);
let plus_one = use_memo(move || double_count() + 1);

// Setting count to 3 triggers both memos
*count.write() = 3;
// double_count() == 6, plus_one() == 7

// Setting count to same doubled value doesn't trigger plus_one
*count.write() = 3; // double_count still 6
// plus_one memo does NOT rerun because double_count value unchanged
```

### 1.5 Async State with `use_resource`

**Use Case:** Fetching data, API calls, async computations

```rust
#[derive(Clone, Debug)]
struct UserData { id: u32, name: String }

async fn fetch_user(user_id: u32) -> Result<UserData, String> {
    // Simulate API call
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    Ok(UserData { id: user_id, name: format!("User {}", user_id) })
}

#[component]
fn UserProfile() -> Element {
    let mut user_id = use_signal(|| 1);

    // Resource reruns when user_id changes
    let user_resource = use_resource(move || async move {
        fetch_user(user_id()).await
    });

    rsx! {
        div {
            // Pattern match on resource state
            match &*user_resource.read_unchecked() {
                Some(Ok(user)) => rsx! { p { "User: {user.name}" } },
                Some(Err(e)) => rsx! { p { "Error: {e}" } },
                None => rsx! { p { "Loading..." } }
            }
            button {
                onclick: move |_| user_id += 1,
                "Next User"
            }
        }
    }
}
```

**Resource States:**
- `None` - Still loading/pending
- `Some(Ok(T))` - Success with data
- `Some(Err(E))` - Error occurred

**CRITICAL:** Avoid holding signal borrows across `.await` points:

```rust
// ❌ BAD: Will panic
use_future(move || async move {
    let mut write = signal.write();
    some_async_work(&mut write).await; // Borrow held over await!
});

// ✅ GOOD: Clone value, do work, then write back
use_future(move || async move {
    let current = signal();
    let result = async_work(current).await;
    signal.set(result);
});
```

---

## 2. Complex Component Composition

### 2.1 Component Props and Reactivity

#### Read-Only Props with `ReadSignal`

```rust
#[component]
fn Meme(caption: ReadOnlySignal<String>) -> Element {
    rsx! {
        div {
            img { src: "/assets/meme.png" }
            h2 { "{caption}" }
        }
    }
}

#[component]
fn MemeEditor() -> Element {
    let mut caption = use_signal(|| "Initial caption".to_string());

    rsx! {
        input {
            value: "{caption}",
            oninput: move |evt| caption.set(evt.value())
        }
        Meme { caption: caption.into() }
    }
}
```

**Benefits:**
- Child component automatically re-renders when signal changes
- No need to manually pass callbacks for updates

#### Mutable Signal Props

```rust
#[component]
fn Child(state: Signal<i32>) -> Element {
    rsx! { "{state}" }
}

fn Parent() -> Element {
    let mut state = use_signal(|| 0);

    rsx! {
        button { onclick: move |_| state += 1, "Increment" }
        Child { state }
    }
}
```

**Key Pattern:** Only the Child component re-renders when state changes, not Parent.

### 2.2 Component Hierarchy Best Practices

#### Split UI into Reusable Components

```rust
// BAD: Monolithic component
#[component]
fn Dashboard() -> Element {
    let mut data = use_signal(|| load_data());

    rsx! {
        div { class: "dashboard",
            // 500+ lines of mixed logic and UI
            div { class: "header", /* ... */ }
            div { class: "sidebar", /* ... */ }
            div { class: "content", /* ... */ }
            div { class: "footer", /* ... */ }
        }
    }
}

// GOOD: Modular components
#[component]
fn Dashboard() -> Element {
    let data = use_signal(|| load_data());

    rsx! {
        div { class: "dashboard",
            DashboardHeader { data: data.into() }
            DashboardSidebar { data: data.into() }
            DashboardContent { data: data.into() }
            DashboardFooter {}
        }
    }
}
```

### 2.3 Event Handling and Data Flow

#### Callbacks via Closures

```rust
#[derive(Props, Clone, PartialEq)]
struct EditorProps {
    on_save: EventHandler<String>,
}

#[component]
fn Editor(props: EditorProps) -> Element {
    let mut text = use_signal(|| String::new());

    rsx! {
        textarea {
            value: "{text}",
            oninput: move |evt| text.set(evt.value())
        }
        button {
            onclick: move |_| props.on_save.call(text()),
            "Save"
        }
    }
}

#[component]
fn ParentComponent() -> Element {
    rsx! {
        Editor {
            on_save: move |text: String| {
                println!("Saved: {}", text);
            }
        }
    }
}
```

---

## 3. Conditional Rendering Patterns

### 3.1 If/Else Expressions

```rust
#[component]
fn LoginPage() -> Element {
    let logged_in = use_signal(|| false);

    rsx! {
        div {
            h1 { "Welcome" }
            if *logged_in.read() {
                div { "You are logged in!" }
            } else {
                div { "Please log in." }
            }
        }
    }
}
```

### 3.2 Match Expressions

**Best for enum-based state:**

```rust
#[derive(Clone, Copy, PartialEq)]
enum LoadingState {
    Idle,
    Loading,
    Success,
    Error,
}

#[component]
fn DataDisplay() -> Element {
    let state = use_signal(|| LoadingState::Idle);

    rsx! {
        div {
            match *state.read() {
                LoadingState::Idle => rsx! { p { "Click to load data" } },
                LoadingState::Loading => rsx! { p { "Loading..." } },
                LoadingState::Success => rsx! { p { "Data loaded!" } },
                LoadingState::Error => rsx! { p { "Error loading data" } },
            }
        }
    }
}
```

### 3.3 Optional Rendering with `if let`

```rust
#[component]
fn UserProfile() -> Element {
    let user = use_signal(|| None::<User>);

    rsx! {
        div {
            if let Some(u) = user.read().as_ref() {
                div {
                    h2 { "{u.name}" }
                    p { "{u.email}" }
                }
            } else {
                p { "No user selected" }
            }
        }
    }
}
```

### 3.4 Conditional Attributes and Classes

```rust
#[component]
fn ToggleButton() -> Element {
    let mut active = use_signal(|| false);

    rsx! {
        button {
            class: if *active.read() { "btn-active" } else { "btn-inactive" },
            onclick: move |_| active.set(!*active.read()),
            if *active.read() { "Active" } else { "Inactive" }
        }
    }
}
```

### 3.5 Reduce Duplication in Conditional Rendering

**Pattern:** Extract common parts, conditionally render only dynamic sections

```rust
// BAD: Duplication
fn LogInApp() -> Element {
    let logged_in = use_signal(|| false);

    if *logged_in.read() {
        rsx! {
            div {
                h1 { "Welcome" }
                p { "Static content" }
                button { "Log out" }
            }
        }
    } else {
        rsx! {
            div {
                h1 { "Welcome" }
                p { "Static content" }
                button { "Log in" }
            }
        }
    }
}

// GOOD: Extract dynamic part
fn LogInImprovedApp() -> Element {
    let logged_in = use_signal(|| false);

    rsx! {
        div {
            h1 { "Welcome" }
            p { "Static content" }
            if *logged_in.read() {
                button { "Log out" }
            } else {
                button { "Log in" }
            }
        }
    }
}
```

---

## 4. Table and Data Grid Patterns

### 4.1 Community Libraries

#### dioxus-table

**Install:**
```toml
[dependencies]
dioxus-table = "0.1"
```

**Features:**
- Derive macro `#[derive(TableData)]` for automatic table generation
- Customizable column rendering
- Sorting and filtering support

**Example:**
```rust
use dioxus_table::*;

#[derive(TableData)]
struct Person {
    #[column(header = "Name")]
    name: String,
    #[column(header = "Age", align = "right")]
    age: u32,
    #[column(header = "Email")]
    email: String,
}

#[component]
fn PeopleTable() -> Element {
    let people = use_signal(|| vec![
        Person { name: "Alice".into(), age: 30, email: "alice@example.com".into() },
        Person { name: "Bob".into(), age: 25, email: "bob@example.com".into() },
    ]);

    rsx! {
        Table { data: people() }
    }
}
```

#### dioxus-sortable

**Features:**
- Sortable table components
- Customizable sort state
- Works with any data type

**Pattern:**
```rust
use dioxus_sortable::*;

#[component]
fn SortableTable() -> Element {
    let mut sort_state = use_signal(|| SortState::default());

    rsx! {
        SortableTable {
            sort_state,
            headers: vec!["Name", "Age", "Score"],
            rows: vec![/* data */]
        }
    }
}
```

### 4.2 Manual Table Implementation

**Best practice for ML results:**

```rust
#[component]
fn PredictionsTable(predictions: Vec<Prediction>) -> Element {
    rsx! {
        div { class: "table-container",
            table { class: "predictions-table",
                thead {
                    tr {
                        th { "Index" }
                        th { "Actual" }
                        th { "Predicted" }
                        th { "Confidence" }
                        th { "Correct" }
                    }
                }
                tbody {
                    for (idx, pred) in predictions.iter().enumerate() {
                        tr {
                            key: "{idx}",
                            class: if pred.is_correct { "correct" } else { "incorrect" },
                            td { "{idx}" }
                            td { "{pred.actual}" }
                            td { "{pred.predicted}" }
                            td { "{pred.confidence:.2}%" }
                            td {
                                if pred.is_correct {
                                    span { class: "icon-check", "✓" }
                                } else {
                                    span { class: "icon-x", "✗" }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
```

**CSS Styling:**
```css
.predictions-table {
    width: 100%;
    border-collapse: collapse;
}

.predictions-table thead {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.predictions-table th {
    padding: 12px;
    text-align: left;
    font-weight: 600;
}

.predictions-table td {
    padding: 10px 12px;
    border-bottom: 1px solid #e9ecef;
}

.predictions-table tr.correct {
    background-color: #d4edda;
}

.predictions-table tr.incorrect {
    background-color: #f8d7da;
}
```

### 4.3 Paginated Tables

```rust
#[component]
fn PaginatedTable(data: Vec<Row>) -> Element {
    let mut page = use_signal(|| 0);
    let page_size = 10;

    let total_pages = (data.len() + page_size - 1) / page_size;
    let start = page() * page_size;
    let end = (start + page_size).min(data.len());
    let page_data = &data[start..end];

    rsx! {
        div {
            table {
                thead { /* headers */ }
                tbody {
                    for (idx, row) in page_data.iter().enumerate() {
                        tr {
                            key: "{start + idx}",
                            td { "{row.value}" }
                        }
                    }
                }
            }
            div { class: "pagination",
                button {
                    disabled: page() == 0,
                    onclick: move |_| page -= 1,
                    "Previous"
                }
                span { "Page {page() + 1} of {total_pages}" }
                button {
                    disabled: page() >= total_pages - 1,
                    onclick: move |_| page += 1,
                    "Next"
                }
            }
        }
    }
}
```

---

## 5. Chart and Visualization Patterns

### 5.1 SVG Charts (Your Current Approach)

**Excellent pattern from your `LossChart` component:**

```rust
#[component]
pub fn LossChart(history: Vec<f64>, current_iteration: usize) -> Element {
    const WIDTH: f64 = 600.0;
    const HEIGHT: f64 = 200.0;
    const PADDING: f64 = 40.0;

    let chart_width = WIDTH - 2.0 * PADDING;
    let chart_height = HEIGHT - 2.0 * PADDING;

    // Filter valid data
    let valid_data: Vec<(usize, f64)> = history
        .iter()
        .enumerate()
        .filter(|(_, &loss)| loss.is_finite())
        .map(|(i, &loss)| (i, loss))
        .collect();

    if valid_data.is_empty() {
        return rsx! { div { class: "chart-empty", "No valid data" } };
    }

    // Calculate bounds
    let min_loss = valid_data.iter().map(|(_, loss)| loss).fold(f64::INFINITY, |a, &b| a.min(b));
    let max_loss = valid_data.iter().map(|(_, loss)| loss).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let loss_range = (max_loss - min_loss).max(1e-10);

    // Build SVG path
    let mut path_data = String::from("M ");
    for (i, (iter, loss)) in valid_data.iter().enumerate() {
        let x = PADDING + (*iter as f64 / history.len() as f64) * chart_width;
        let y = PADDING + chart_height - ((loss - min_loss) / loss_range * chart_height);

        if i == 0 {
            path_data.push_str(&format!("{} {}", x, y));
        } else {
            path_data.push_str(&format!(" L {} {}", x, y));
        }
    }

    rsx! {
        svg {
            class: "loss-chart",
            width: "{WIDTH}",
            height: "{HEIGHT}",
            view_box: "0 0 {WIDTH} {HEIGHT}",

            // Grid lines
            for i in 0..5 {
                {
                    let y = PADDING + (i as f64 / 4.0) * chart_height;
                    rsx! {
                        line {
                            x1: "{PADDING}",
                            y1: "{y}",
                            x2: "{PADDING + chart_width}",
                            y2: "{y}",
                            stroke: "#e9ecef",
                            stroke_width: "1",
                        }
                    }
                }
            }

            // Data path
            path {
                d: "{path_data}",
                fill: "none",
                stroke: "url(#gradient)",
                stroke_width: "3",
            }

            // Gradient definition
            defs {
                linearGradient {
                    id: "gradient",
                    x1: "0%", y1: "0%", x2: "100%", y2: "0%",
                    stop { offset: "0%", stop_color: "#6c5ce7" }
                    stop { offset: "100%", stop_color: "#a29bfe" }
                }
            }
        }
    }
}
```

### 5.2 Bar Chart Pattern

```rust
#[component]
fn BarChart(data: Vec<(String, f64)>) -> Element {
    const WIDTH: f64 = 600.0;
    const HEIGHT: f64 = 400.0;
    const PADDING: f64 = 50.0;

    let max_value = data.iter().map(|(_, v)| v).fold(0.0, |a, &b| a.max(b));
    let bar_width = (WIDTH - 2.0 * PADDING) / data.len() as f64 * 0.8;
    let chart_height = HEIGHT - 2.0 * PADDING;

    rsx! {
        svg {
            width: "{WIDTH}",
            height: "{HEIGHT}",

            for (i, (label, value)) in data.iter().enumerate() {
                {
                    let x = PADDING + i as f64 * (WIDTH - 2.0 * PADDING) / data.len() as f64;
                    let bar_height = (value / max_value) * chart_height;
                    let y = HEIGHT - PADDING - bar_height;

                    rsx! {
                        // Bar
                        rect {
                            x: "{x}",
                            y: "{y}",
                            width: "{bar_width}",
                            height: "{bar_height}",
                            fill: "#667eea",
                            rx: "4",
                        }
                        // Label
                        text {
                            x: "{x + bar_width / 2.0}",
                            y: "{HEIGHT - PADDING + 20.0}",
                            text_anchor: "middle",
                            font_size: "12",
                            "{label}"
                        }
                        // Value
                        text {
                            x: "{x + bar_width / 2.0}",
                            y: "{y - 5.0}",
                            text_anchor: "middle",
                            font_size: "10",
                            fill: "#666",
                            "{value:.2}"
                        }
                    }
                }
            }
        }
    }
}
```

### 5.3 Community Chart Libraries

#### dioxus-charts

**Repository:** https://github.com/dioxus-community/dioxus-charts

**Features:**
- Line charts
- Bar charts
- Pie charts
- Simple API

**Example:**
```rust
use dioxus_charts::*;

#[component]
fn ChartExample() -> Element {
    let data = vec![
        DataPoint { x: 0.0, y: 1.0 },
        DataPoint { x: 1.0, y: 2.5 },
        DataPoint { x: 2.0, y: 1.8 },
    ];

    rsx! {
        LineChart {
            data,
            width: 600,
            height: 400,
        }
    }
}
```

---

## 6. Tab and Accordion Patterns

### 6.1 Tab Component Pattern

```rust
#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Overview,
    Details,
    Settings,
}

impl Tab {
    fn label(&self) -> &'static str {
        match self {
            Tab::Overview => "Overview",
            Tab::Details => "Details",
            Tab::Settings => "Settings",
        }
    }
}

#[component]
fn TabbedInterface() -> Element {
    let mut active_tab = use_signal(|| Tab::Overview);

    rsx! {
        div { class: "tabbed-interface",
            // Tab headers
            div { class: "tab-headers",
                for tab in [Tab::Overview, Tab::Details, Tab::Settings] {
                    button {
                        class: if *active_tab.read() == tab { "tab-active" } else { "tab-inactive" },
                        onclick: move |_| active_tab.set(tab),
                        "{tab.label()}"
                    }
                }
            }

            // Tab content
            div { class: "tab-content",
                match *active_tab.read() {
                    Tab::Overview => rsx! { OverviewPanel {} },
                    Tab::Details => rsx! { DetailsPanel {} },
                    Tab::Settings => rsx! { SettingsPanel {} },
                }
            }
        }
    }
}
```

**CSS:**
```css
.tab-headers {
    display: flex;
    gap: 4px;
    border-bottom: 2px solid #e9ecef;
}

.tab-headers button {
    padding: 12px 24px;
    border: none;
    background: none;
    cursor: pointer;
    font-size: 16px;
    border-bottom: 3px solid transparent;
    transition: all 0.3s ease;
}

.tab-active {
    color: #667eea;
    border-bottom-color: #667eea !important;
    font-weight: 600;
}

.tab-inactive:hover {
    background-color: #f8f9fa;
}

.tab-content {
    padding: 24px;
    animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}
```

### 6.2 Accordion Component

```rust
#[component]
fn AccordionItem(
    title: String,
    is_open: Signal<bool>,
    children: Element,
) -> Element {
    let mut is_open_mut = is_open;

    rsx! {
        div { class: "accordion-item",
            button {
                class: "accordion-header",
                onclick: move |_| is_open_mut.set(!*is_open_mut.read()),
                aria_expanded: "{is_open()}",

                span { "{title}" }
                span { class: "accordion-icon",
                    if *is_open.read() { "▼" } else { "▶" }
                }
            }

            if *is_open.read() {
                div {
                    class: "accordion-content",
                    {children}
                }
            }
        }
    }
}

#[component]
fn Accordion() -> Element {
    let mut section1_open = use_signal(|| false);
    let mut section2_open = use_signal(|| false);
    let mut section3_open = use_signal(|| false);

    rsx! {
        div { class: "accordion",
            AccordionItem {
                title: "Section 1",
                is_open: section1_open,
                div { "Content for section 1" }
            }
            AccordionItem {
                title: "Section 2",
                is_open: section2_open,
                div { "Content for section 2" }
            }
            AccordionItem {
                title: "Section 3",
                is_open: section3_open,
                div { "Content for section 3" }
            }
        }
    }
}
```

**Accessibility Considerations:**
- Use `aria-expanded` attribute
- Ensure keyboard navigation with `tabindex="0"`
- Add screen reader support with ARIA labels

### 6.3 Dioxus Primitives (Official Component Library)

**Repository:** https://github.com/DioxusLabs/components

**Features:**
- 28/29 ARIA-accessible components
- Unstyled (bring your own CSS)
- Based on Radix Primitives
- Includes: Accordion, Tabs, Dialog, Popover, etc.

**Example Usage:**
```rust
use dioxus_primitives::*;

#[component]
fn MyAccordion() -> Element {
    rsx! {
        Accordion {
            AccordionItem {
                value: "item-1",
                AccordionTrigger { "Section 1" }
                AccordionContent { "Content 1" }
            }
            AccordionItem {
                value: "item-2",
                AccordionTrigger { "Section 2" }
                AccordionContent { "Content 2" }
            }
        }
    }
}
```

---

## 7. Performance Optimization

### 7.1 Minimize Dynamic RSX

**Key Principle:** Static parts are skipped during diffing

```rust
// BAD: Everything is dynamic
#[component]
fn SlowComponent() -> Element {
    let data = use_signal(|| vec![1, 2, 3]);

    rsx! {
        div {
            for item in data.read().iter() {
                div {
                    h2 { "Item" }  // Wasteful: re-creates on every render
                    p { "{item}" }
                }
            }
        }
    }
}

// GOOD: Static parts are cached
#[component]
fn FastComponent() -> Element {
    let data = use_signal(|| vec![1, 2, 3]);

    rsx! {
        div {
            h2 { "Items" }  // Static: rendered once
            for item in data.read().iter() {
                p { key: "{item}", "{item}" }  // Only dynamic part
            }
        }
    }
}
```

### 7.2 Virtualization for Large Lists

**Library:** `dioxus-lazy`

**Use Case:** Rendering 1000+ items efficiently

```rust
use dioxus_lazy::*;

#[component]
fn LargeList() -> Element {
    let items = use_signal(|| (0..10000).collect::<Vec<_>>());

    rsx! {
        VirtualList {
            len: items.read().len(),
            size: 600,  // Container height
            item_size: 50,  // Each item height
            builder: move |idx| {
                let item = items.read()[idx];
                rsx! {
                    div { class: "list-item",
                        "Item {item}"
                    }
                }
            }
        }
    }
}
```

**How it works:**
- Only renders visible items + buffer
- Dynamically adds/removes items as user scrolls
- Essential for 1000+ row tables

### 7.3 Memoization for Expensive Components

```rust
#[component]
fn ExpensiveChild(data: Vec<f64>) -> Element {
    // This component only re-renders if data changes
    rsx! {
        div {
            "Computed result: {data.iter().sum::<f64>()}"
        }
    }
}

#[component]
fn Parent() -> Element {
    let mut count = use_signal(|| 0);
    let data = use_signal(|| vec![1.0, 2.0, 3.0]);

    rsx! {
        button { onclick: move |_| count += 1, "Count: {count}" }

        // ExpensiveChild only re-renders when data changes, NOT when count changes
        ExpensiveChild { data: data() }
    }
}
```

### 7.4 Avoid Unnecessary State Updates During Render

```rust
// ❌ ANTIPATTERN: Infinite loop
#[component]
fn BadCounter() -> Element {
    let mut count = use_signal(|| 0);

    // This runs on EVERY render, causing infinite re-renders
    count.set(*count.read() + 1);

    rsx! { div { "Count: {count}" } }
}

// ✅ CORRECT: Update in event handler
#[component]
fn GoodCounter() -> Element {
    let mut count = use_signal(|| 0);

    rsx! {
        div { "Count: {count}" }
        button {
            onclick: move |_| count.set(*count.read() + 1),
            "Increment"
        }
    }
}
```

### 7.5 Performance Monitoring

**Browser DevTools:**
- Chrome Performance tab: Record 10s session, check FPS
- Memory tab: Heap snapshots to detect leaks
- Network tab: Check WASM bundle size (<2MB ideal)

**Profiling Pattern:**
```javascript
// In browser console
const start = performance.now();
// Run operation (e.g., click button, load data)
const elapsed = performance.now() - start;
console.log(`Operation took ${elapsed}ms`);
```

---

## 8. Styling and CSS Integration

### 8.1 Class-Based Styling (Recommended)

**Pattern from your codebase:**

```rust
#[component]
fn StyledComponent() -> Element {
    rsx! {
        div { class: "model-performance-card",
            div { class: "perf-header",
                h3 { "Title" }
                span { class: "status-badge", "Active" }
            }
            div { class: "metrics-grid",
                div { class: "metric-card loss",
                    span { class: "metric-label", "Loss" }
                    span { class: "metric-value", "0.123" }
                }
            }
        }
    }
}
```

**CSS (main.css):**
```css
.model-performance-card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.perf-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.status-badge {
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    color: white;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
}

.metric-card {
    display: flex;
    flex-direction: column;
    padding: 16px;
    border-radius: 8px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.metric-label {
    font-size: 14px;
    opacity: 0.9;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 24px;
    font-weight: 700;
}
```

### 8.2 Dynamic Inline Styles

```rust
#[component]
fn ProgressBar(percent: f64) -> Element {
    let color = if percent < 30.0 {
        "#e74c3c"
    } else if percent < 70.0 {
        "#f39c12"
    } else {
        "#27ae60"
    };

    rsx! {
        div { class: "progress-container",
            div {
                class: "progress-bar",
                style: "width: {percent}%; background-color: {color}",
            }
        }
    }
}
```

### 8.3 Conditional Classes

```rust
#[component]
fn Button(active: bool) -> Element {
    let classes = if active {
        "btn btn-active"
    } else {
        "btn btn-inactive"
    };

    rsx! {
        button { class: "{classes}", "Click me" }
    }
}

// Alternative: String building
#[component]
fn FlexibleButton(variant: &str, disabled: bool) -> Element {
    let classes = format!(
        "btn btn-{} {}",
        variant,
        if disabled { "disabled" } else { "" }
    );

    rsx! {
        button { class: "{classes}", "Submit" }
    }
}
```

### 8.4 CSS Grid and Flexbox Patterns

**Grid Layout:**
```css
.dashboard-grid {
    display: grid;
    grid-template-columns: 250px 1fr;
    grid-template-rows: auto 1fr auto;
    grid-template-areas:
        "header header"
        "sidebar content"
        "footer footer";
    gap: 20px;
    min-height: 100vh;
}

.dashboard-header { grid-area: header; }
.dashboard-sidebar { grid-area: sidebar; }
.dashboard-content { grid-area: content; }
.dashboard-footer { grid-area: footer; }
```

**Flexbox Layout:**
```css
.metrics-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}

.metric-card {
    flex: 1 1 200px;  /* Grow, shrink, min-width */
    min-width: 200px;
    max-width: 300px;
}
```

### 8.5 Responsive Design

```css
/* Mobile-first approach */
.container {
    padding: 16px;
}

/* Tablet */
@media (min-width: 768px) {
    .container {
        padding: 24px;
    }

    .metrics-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

/* Desktop */
@media (min-width: 1024px) {
    .container {
        max-width: 1280px;
        margin: 0 auto;
        padding: 32px;
    }

    .metrics-grid {
        grid-template-columns: repeat(4, 1fr);
    }
}
```

---

## 9. Recommended Patterns for ML Playground

### 9.1 Result Display Architecture

**Recommended Structure:**

```
MLPlayground
├── DataUpload (existing)
├── AlgorithmSelector (existing)
├── AlgorithmConfigurator (existing)
└── ResultsDisplay (NEW)
    ├── ResultTabs
    │   ├── OverviewTab
    │   ├── DetailsTab
    │   └── VisualizationTab
    ├── PerformanceMetrics (existing)
    └── PredictionsTable (NEW)
```

### 9.2 Tabbed Results Interface

```rust
#[derive(Clone, Copy, PartialEq)]
enum ResultTab {
    Overview,
    Predictions,
    Metrics,
}

#[component]
fn ResultsDisplay(
    algorithm: AlgorithmType,
    results: AlgorithmResults,
    metrics: PerformanceMetrics,
) -> Element {
    let mut active_tab = use_signal(|| ResultTab::Overview);

    rsx! {
        div { class: "results-container",
            // Tab navigation
            div { class: "result-tabs",
                for tab in [ResultTab::Overview, ResultTab::Predictions, ResultTab::Metrics] {
                    button {
                        class: if *active_tab.read() == tab { "tab-active" } else { "tab" },
                        onclick: move |_| active_tab.set(tab),
                        {match tab {
                            ResultTab::Overview => "📊 Overview",
                            ResultTab::Predictions => "🎯 Predictions",
                            ResultTab::Metrics => "📈 Metrics",
                        }}
                    }
                }
            }

            // Tab content
            div { class: "tab-content",
                match *active_tab.read() {
                    ResultTab::Overview => rsx! {
                        OverviewPanel { algorithm, results: results.clone() }
                    },
                    ResultTab::Predictions => rsx! {
                        PredictionsTable { predictions: results.predictions.clone() }
                    },
                    ResultTab::Metrics => rsx! {
                        ModelPerformanceCard { metrics: metrics.clone() }
                    },
                }
            }
        }
    }
}
```

### 9.3 Algorithm-Specific Result Panels

```rust
#[component]
fn OverviewPanel(algorithm: AlgorithmType, results: AlgorithmResults) -> Element {
    rsx! {
        div { class: "overview-panel",
            match algorithm {
                AlgorithmType::KMeans => rsx! {
                    ClusteringResults {
                        clusters: results.cluster_assignments,
                        centers: results.cluster_centers,
                    }
                },
                AlgorithmType::LogisticRegression => rsx! {
                    ClassificationResults {
                        accuracy: results.accuracy,
                        confusion_matrix: results.confusion_matrix,
                        coefficients: results.coefficients,
                    }
                },
                AlgorithmType::DecisionTree => rsx! {
                    TreeResults {
                        tree_depth: results.tree_depth,
                        feature_importance: results.feature_importance,
                        accuracy: results.accuracy,
                    }
                },
                _ => rsx! { div { "Results for {algorithm.name()}" } }
            }
        }
    }
}
```

### 9.4 Confusion Matrix Visualization

```rust
#[component]
fn ConfusionMatrix(matrix: Vec<Vec<usize>>, labels: Vec<String>) -> Element {
    let n_classes = matrix.len();
    let max_value = matrix.iter()
        .flat_map(|row| row.iter())
        .max()
        .unwrap_or(&1);

    rsx! {
        div { class: "confusion-matrix",
            h3 { "Confusion Matrix" }
            table { class: "cm-table",
                thead {
                    tr {
                        th {} // Empty corner
                        th { colspan: "{n_classes}", "Predicted" }
                    }
                    tr {
                        th {} // Empty corner
                        for label in labels.iter() {
                            th { "{label}" }
                        }
                    }
                }
                tbody {
                    for (i, row) in matrix.iter().enumerate() {
                        tr {
                            if i == 0 {
                                th { rowspan: "{n_classes}", class: "row-header", "Actual" }
                            }
                            th { "{labels[i]}" }

                            for (j, &count) in row.iter().enumerate() {
                                {
                                    let intensity = (count as f64 / *max_value as f64 * 255.0) as u8;
                                    let bg_color = if i == j {
                                        format!("rgba(46, 213, 115, {})", intensity as f64 / 255.0)
                                    } else {
                                        format!("rgba(255, 107, 107, {})", intensity as f64 / 255.0)
                                    };

                                    rsx! {
                                        td {
                                            class: "cm-cell",
                                            style: "background-color: {bg_color}",
                                            "{count}"
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
}
```

**CSS:**
```css
.confusion-matrix {
    margin: 20px 0;
}

.cm-table {
    border-collapse: collapse;
    margin: 0 auto;
}

.cm-table th {
    background: #667eea;
    color: white;
    padding: 12px;
    font-weight: 600;
}

.cm-cell {
    padding: 16px;
    text-align: center;
    font-weight: 700;
    min-width: 60px;
    border: 1px solid #e9ecef;
}

.row-header {
    writing-mode: vertical-rl;
    transform: rotate(180deg);
}
```

### 9.5 Feature Importance Chart

```rust
#[component]
fn FeatureImportanceChart(importances: Vec<(String, f64)>) -> Element {
    let max_importance = importances.iter()
        .map(|(_, imp)| imp)
        .fold(0.0, |a, &b| a.max(b));

    rsx! {
        div { class: "feature-importance",
            h3 { "Feature Importance" }
            div { class: "importance-bars",
                for (feature, importance) in importances.iter() {
                    {
                        let width_pct = (importance / max_importance * 100.0).min(100.0);
                        rsx! {
                            div { class: "importance-row",
                                span { class: "feature-name", "{feature}" }
                                div { class: "importance-bar-container",
                                    div {
                                        class: "importance-bar",
                                        style: "width: {width_pct}%",
                                    }
                                    span { class: "importance-value", "{importance:.3}" }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
```

**CSS:**
```css
.feature-importance {
    margin: 20px 0;
}

.importance-row {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
    gap: 12px;
}

.feature-name {
    flex: 0 0 150px;
    font-weight: 600;
    text-align: right;
}

.importance-bar-container {
    flex: 1;
    position: relative;
    height: 30px;
    background: #f8f9fa;
    border-radius: 4px;
    overflow: hidden;
}

.importance-bar {
    height: 100%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    transition: width 0.5s ease;
}

.importance-value {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 12px;
    font-weight: 700;
    color: white;
    mix-blend-mode: difference;
}
```

### 9.6 Paginated Predictions Table with Search

```rust
#[component]
fn PredictionsTable(predictions: Vec<Prediction>) -> Element {
    let mut page = use_signal(|| 0);
    let mut search = use_signal(|| String::new());
    let page_size = 50;

    // Filter predictions based on search
    let filtered: Vec<&Prediction> = predictions.iter()
        .filter(|p| {
            let query = search().to_lowercase();
            query.is_empty() ||
            p.actual.to_string().contains(&query) ||
            p.predicted.to_string().contains(&query)
        })
        .collect();

    let total_pages = (filtered.len() + page_size - 1) / page_size;
    let start = page() * page_size;
    let end = (start + page_size).min(filtered.len());
    let page_data = &filtered[start..end];

    rsx! {
        div { class: "predictions-container",
            // Search bar
            div { class: "search-bar",
                input {
                    r#type: "text",
                    placeholder: "Search predictions...",
                    value: "{search}",
                    oninput: move |evt| {
                        search.set(evt.value());
                        page.set(0); // Reset to first page
                    }
                }
                span { class: "result-count", "{filtered.len()} results" }
            }

            // Table
            table { class: "predictions-table",
                thead {
                    tr {
                        th { "Index" }
                        th { "Actual" }
                        th { "Predicted" }
                        th { "Confidence" }
                        th { "Status" }
                    }
                }
                tbody {
                    for (i, pred) in page_data.iter().enumerate() {
                        tr {
                            key: "{start + i}",
                            class: if pred.is_correct { "correct" } else { "incorrect" },
                            td { "{start + i + 1}" }
                            td { "{pred.actual}" }
                            td { "{pred.predicted}" }
                            td {
                                div { class: "confidence-bar",
                                    div {
                                        class: "confidence-fill",
                                        style: "width: {pred.confidence}%",
                                    }
                                    span { "{pred.confidence:.1}%" }
                                }
                            }
                            td {
                                if pred.is_correct {
                                    span { class: "status-correct", "✓ Correct" }
                                } else {
                                    span { class: "status-incorrect", "✗ Wrong" }
                                }
                            }
                        }
                    }
                }
            }

            // Pagination
            div { class: "pagination",
                button {
                    disabled: page() == 0,
                    onclick: move |_| page -= 1,
                    "← Previous"
                }
                span { class: "page-info",
                    "Page {page() + 1} of {total_pages}"
                }
                button {
                    disabled: page() >= total_pages - 1,
                    onclick: move |_| page += 1,
                    "Next →"
                }
            }
        }
    }
}
```

### 9.7 Collapsible Results Sections (Accordion)

```rust
#[component]
fn CollapsibleResults(results: AlgorithmResults) -> Element {
    let mut show_summary = use_signal(|| true);
    let mut show_details = use_signal(|| false);
    let mut show_diagnostics = use_signal(|| false);

    rsx! {
        div { class: "collapsible-results",
            // Summary section (always visible by default)
            CollapsibleSection {
                title: "📊 Summary",
                is_open: show_summary,
                div {
                    p { "Accuracy: {results.accuracy:.2}%" }
                    p { "Total predictions: {results.total}" }
                    p { "Correct: {results.correct}" }
                    p { "Incorrect: {results.incorrect}" }
                }
            }

            // Detailed metrics (collapsed by default)
            CollapsibleSection {
                title: "📈 Detailed Metrics",
                is_open: show_details,
                div {
                    DetailedMetricsTable { metrics: results.detailed_metrics.clone() }
                }
            }

            // Diagnostics (collapsed by default)
            CollapsibleSection {
                title: "🔍 Diagnostics",
                is_open: show_diagnostics,
                div {
                    DiagnosticsPanel { diagnostics: results.diagnostics.clone() }
                }
            }
        }
    }
}

#[component]
fn CollapsibleSection(
    title: String,
    is_open: Signal<bool>,
    children: Element,
) -> Element {
    let mut is_open_mut = is_open;

    rsx! {
        div { class: "collapsible-section",
            button {
                class: "section-header",
                onclick: move |_| is_open_mut.set(!*is_open_mut.read()),

                span { "{title}" }
                span { class: "toggle-icon",
                    if *is_open.read() { "▼" } else { "▶" }
                }
            }

            if *is_open.read() {
                div {
                    class: "section-content",
                    {children}
                }
            }
        }
    }
}
```

---

## Summary: Best Practices for ML Playground

### State Management
1. **Use `use_signal`** for all reactive state
2. **Use `use_memo`** for derived/computed values (e.g., accuracy, statistics)
3. **Use `use_resource`** for async data loading (if fetching from APIs in future)
4. **Global signals** for app-wide settings (dark mode, preferences)

### Component Structure
1. **Break down into small components** (<200 lines each)
2. **Use props with `ReadSignal`** for reactive data flow
3. **Extract reusable patterns** (tables, charts, cards) into separate components
4. **Match algorithm types** with conditional rendering for algorithm-specific displays

### Rendering Patterns
1. **Tabs for multiple views** (Overview, Predictions, Metrics)
2. **Accordions for optional details** (Diagnostics, Advanced Metrics)
3. **Tables with pagination** for large result sets (50+ rows)
4. **SVG for charts** (performance is good enough for <1000 points)
5. **Conditional classes** for visual feedback (correct/incorrect predictions)

### Performance
1. **Minimize dynamic RSX** - keep static parts outside loops
2. **Use `key` attribute** for all list items
3. **Virtualize large lists** with `dioxus-lazy` (1000+ items)
4. **Memoize expensive computations** with `use_memo`
5. **Avoid state updates during render** - only in event handlers

### Styling
1. **Class-based CSS** in `main.css`
2. **CSS Grid** for layout (dashboard, metrics grid)
3. **Flexbox** for rows and dynamic sizing
4. **Gradients and colors** matching your purple/blue theme
5. **Animations** for smooth transitions (fade-in, slide)

### Accessibility
1. **ARIA attributes** for interactive elements (`aria-expanded`, `aria-label`)
2. **Semantic HTML** (tables, headers, sections)
3. **Keyboard navigation** (tab order, focus states)
4. **Screen reader support** (descriptive labels)

---

## Next Steps

1. **Implement Tabbed Results Interface**
   - Create `ResultTab` enum
   - Build `ResultsDisplay` component
   - Add tab navigation with CSS transitions

2. **Build Predictions Table**
   - Pagination (50 rows per page)
   - Search/filter functionality
   - Color-coded correct/incorrect
   - Confidence visualization

3. **Add Confusion Matrix**
   - SVG heatmap visualization
   - Color intensity based on counts
   - Interactive tooltips (future enhancement)

4. **Feature Importance Chart**
   - Horizontal bar chart
   - Sorted by importance
   - Algorithm-specific (Decision Tree, LogReg)

5. **Performance Optimization**
   - Profile large datasets (1000+ rows)
   - Consider virtualization if needed
   - Optimize SVG rendering for multiple charts

---

**Key Files to Create:**

- `/web/src/components/ml_playground/results_display.rs` - Main results container
- `/web/src/components/ml_playground/predictions_table.rs` - Paginated table
- `/web/src/components/ml_playground/confusion_matrix.rs` - Classification matrix
- `/web/src/components/ml_playground/feature_importance.rs` - Importance chart
- `/web/src/components/ml_playground/result_tabs.rs` - Tab navigation

**CSS Additions to `main.css`:**

```css
/* Tab styles */
.result-tabs { /* ... */ }
.tab-active { /* ... */ }
.tab-content { /* ... */ }

/* Table styles */
.predictions-table { /* ... */ }
.predictions-table tr.correct { /* ... */ }
.predictions-table tr.incorrect { /* ... */ }

/* Chart styles */
.confusion-matrix { /* ... */ }
.feature-importance { /* ... */ }
.importance-bar { /* ... */ }

/* Pagination */
.pagination { /* ... */ }
```

---

**References:**

- Dioxus Official Docs: https://dioxuslabs.com/learn/0.6/
- Dioxus Components: https://github.com/DioxusLabs/components
- dioxus-table: https://github.com/Synphonyte/dioxus-table
- dioxus-lazy: https://github.com/dioxus-community/dioxus-lazy
- dioxus-charts: https://github.com/dioxus-community/dioxus-charts

---

**End of Research Document**
