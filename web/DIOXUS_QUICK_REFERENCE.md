# Dioxus 0.6 Quick Reference Guide

A concise reference for common Dioxus patterns and APIs.

## Core Hooks

### State Management

```rust
// Local reactive state
let mut count = use_signal(|| 0);
count += 1;                          // Increment
count.set(10);                       // Set value
let value = count();                 // Read value
count.with(|val| println!("{}", val)); // Read with closure

// Derived state (memoized)
let double = use_memo(move || count() * 2);

// Async state
let data = use_resource(move || async move {
    fetch_data().await
});

// Side effects
use_effect(move || {
    println!("Count changed: {}", count());
});
```

### Context

```rust
// Provide context
use_context_provider(|| Signal::new(AppState::default()));

// Consume context
let state = use_context::<Signal<AppState>>();
```

### Coroutines

```rust
let task = use_coroutine(|mut rx: UnboundedReceiver<Action>| async move {
    while let Some(action) = rx.next().await {
        // Handle action
    }
});

task.send(Action::Start);
```

---

## RSX Syntax

### Basic Elements

```rust
rsx! {
    div { class: "container",
        h1 { "Title" }
        p { "Paragraph with {variable}" }
        button { onclick: |_| println!("Clicked"), "Click me" }
    }
}
```

### Conditionals

```rust
rsx! {
    // If expression
    if condition {
        p { "True" }
    }

    // If-else
    if condition {
        p { "True" }
    } else {
        p { "False" }
    }

    // Match
    match value {
        Some(x) => rsx! { p { "{x}" } },
        None => rsx! { p { "None" } }
    }
}
```

### Loops

```rust
rsx! {
    // For loop
    for item in items.iter() {
        li { "{item}" }
    }

    // With key
    for (id, item) in items.iter().enumerate() {
        li { key: "{id}", "{item}" }
    }
}
```

### Event Handlers

```rust
rsx! {
    // Mouse events
    div {
        onclick: move |evt| println!("Click: {:?}", evt.page_coordinates()),
        onmouseenter: move |_| hovered.set(true),
        onmouseleave: move |_| hovered.set(false),
        onmousemove: move |evt| position.set(evt.page_coordinates()),
    }

    // Keyboard events
    input {
        onkeydown: move |evt| {
            match evt.key().as_str() {
                "Enter" => submit(),
                _ => {}
            }
        },
    }

    // Form events
    input {
        value: "{text()}",
        oninput: move |evt| text.set(evt.value()),
    }
}
```

---

## SVG

### Basic Shapes

```rust
rsx! {
    svg { width: "400", height: "400",
        // Circle
        circle { cx: "200", cy: "200", r: "50", fill: "blue" }

        // Rectangle
        rect { x: "10", y: "10", width: "100", height: "50", fill: "red" }

        // Line
        line { x1: "0", y1: "0", x2: "100", y2: "100", stroke: "black" }

        // Polyline
        polyline {
            points: "0,0 50,50 100,0",
            fill: "none",
            stroke: "green"
        }

        // Path
        path {
            d: "M 10 10 L 50 50 L 90 10",
            fill: "none",
            stroke: "purple"
        }

        // Text
        text { x: "50", y: "50", "Label" }
    }
}
```

### Grouping and Transformations

```rust
rsx! {
    svg { width: "400", height: "400",
        g { class: "layer", transform: "translate(50, 50)",
            circle { r: "20", fill: "blue" }
        }

        g { transform: "rotate(45, 200, 200)",
            rect { width: "50", height: "50", fill: "red" }
        }
    }
}
```

---

## Component Patterns

### Simple Component

```rust
#[component]
fn Greeting(name: String) -> Element {
    rsx! {
        p { "Hello, {name}!" }
    }
}

// Usage
rsx! {
    Greeting { name: "World".into() }
}
```

### Component with Signal Props

```rust
#[component]
fn Counter(count: Signal<i32>) -> Element {
    rsx! {
        button {
            onclick: move |_| count += 1,
            "Count: {count}"
        }
    }
}
```

### Component with Children

```rust
#[component]
fn Card(children: Element) -> Element {
    rsx! {
        div { class: "card",
            {children}
        }
    }
}

// Usage
rsx! {
    Card {
        p { "Content" }
    }
}
```

---

## Router

### Setup

```rust
#[derive(Routable, Clone, PartialEq)]
enum Route {
    #[route("/")]
    Home,

    #[route("/about")]
    About,

    #[route("/user/:id")]
    User { id: i32 },
}

fn App() -> Element {
    rsx! {
        Router::<Route> {}
    }
}
```

### Navigation

```rust
fn NavBar() -> Element {
    rsx! {
        nav {
            Link { to: Route::Home, "Home" }
            Link { to: Route::About, "About" }
            Link { to: Route::User { id: 1 }, "User 1" }
        }
    }
}
```

### Route Components

```rust
#[component]
fn Home() -> Element {
    rsx! { h1 { "Home" } }
}

#[component]
fn User(id: i32) -> Element {
    rsx! { h1 { "User {id}" } }
}
```

---

## Async Operations

### Spawning Tasks

```rust
// Spawn future
spawn(async move {
    let result = fetch_data().await;
    data.set(result);
});

// Spawn blocking task
spawn(async move {
    let result = tokio::task::spawn_blocking(|| {
        expensive_computation()
    }).await;
});
```

### use_resource Pattern

```rust
let data = use_resource(move || async move {
    api_call().await
});

rsx! {
    match &*data.read() {
        Some(Ok(value)) => rsx! { p { "{value}" } },
        Some(Err(e)) => rsx! { p { "Error: {e}" } },
        None => rsx! { p { "Loading..." } }
    }
}
```

---

## JavaScript Interop

### use_eval

```rust
let result = use_signal(|| String::new());

let run_js = move |_| {
    let eval = eval(r#"
        const x = 2 + 2;
        dioxus.send(x.toString());
    "#);

    spawn(async move {
        if let Ok(msg) = eval.recv().await {
            result.set(msg.to_string());
        }
    });
};
```

### wasm-bindgen

```rust
use wasm_bindgen::prelude::*;
use web_sys::console;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

fn component() -> Element {
    rsx! {
        button {
            onclick: move |_| {
                alert("Hello!");
                console::log_1(&"Clicked".into());
            },
            "Alert"
        }
    }
}
```

---

## Styling

### Inline Styles

```rust
rsx! {
    div {
        style: "color: red; font-size: 20px;",
        "Styled text"
    }
}
```

### CSS Classes

```rust
rsx! {
    div {
        class: "card primary",
        class: if active() { "active" } else { "" },
        "Content"
    }
}
```

### Asset Management

```rust
static CSS: Asset = asset!("/assets/main.css");

fn App() -> Element {
    rsx! {
        document::Stylesheet { href: CSS }
        // Rest of app
    }
}
```

---

## Performance Optimization

### Keys for Lists

```rust
// Good: with keys
for (id, item) in items.iter().enumerate() {
    Item { key: "{id}", item }
}

// Bad: without keys (re-renders all items)
for item in items.iter() {
    Item { item }
}
```

### Memoization

```rust
// Expensive computation only runs when deps change
let result = use_memo(move || {
    expensive_computation(input())
});
```

### Avoid Holding Borrows

```rust
// Bad: panics
let value = signal.read();
signal += 1;  // PANIC

// Good: scoped
signal.with(|val| {
    use_value(val);
});
signal += 1;  // OK
```

---

## Common Patterns

### Toggle State

```rust
let mut visible = use_signal(|| false);

rsx! {
    button {
        onclick: move |_| visible.toggle(),
        "Toggle"
    }
}
```

### Form Input

```rust
let mut text = use_signal(String::new);

rsx! {
    input {
        value: "{text()}",
        oninput: move |evt| text.set(evt.value()),
    }
    p { "You typed: {text}" }
}
```

### Debounced Input

```rust
let mut input = use_signal(String::new);
let mut debounced = use_signal(String::new);

use_coroutine(move |_: UnboundedReceiver<()>| async move {
    let mut last_value = String::new();

    loop {
        tokio::time::sleep(Duration::from_millis(300)).await;

        let current = input();
        if current != last_value {
            debounced.set(current.clone());
            last_value = current;
        }
    }
});
```

### Loading State

```rust
let mut loading = use_signal(|| false);

let fetch_data = move |_| {
    loading.set(true);

    spawn(async move {
        let result = api_call().await;
        data.set(result);
        loading.set(false);
    });
};

rsx! {
    if loading() {
        p { "Loading..." }
    } else {
        p { "Data: {data()}" }
    }
}
```

---

## Cargo.toml Configuration

### Web Target

```toml
[dependencies]
dioxus = { version = "0.6", features = ["router", "web"] }

[profile.release]
opt-level = "z"
lto = true
codegen-units = 1
strip = true
```

### Desktop Target

```toml
[dependencies]
dioxus = { version = "0.6", features = ["router", "desktop"] }
```

### Full-Stack

```toml
[dependencies]
dioxus = { version = "0.6", features = ["router", "fullstack"] }
dioxus-fullstack = "0.6"
```

---

## CLI Commands

```bash
# Development server
dx serve

# Build for production
dx build --release

# Desktop app
dx serve --platform desktop

# Mobile
dx serve --platform mobile

# Custom port
dx serve --port 3000

# Hot reload enabled by default
```

---

## Debugging

### Console Logging

```rust
use web_sys::console;

console::log_1(&"Message".into());
console::log_2(&"Key:".into(), &value.into());
console::error_1(&"Error".into());
```

### Debug Print

```rust
use_effect(move || {
    println!("State changed: {:?}", state());
});
```

### Browser DevTools

- React DevTools integration (experimental)
- Standard browser console
- Network tab for API calls
- Performance profiling

---

## Common Errors & Solutions

### "Cannot borrow as mutable"

```rust
// Problem: trying to mutate in read context
let value = signal.read();
signal += 1;  // Error

// Solution: use separate scopes
{
    let value = signal.read();
    // use value
}
signal += 1;  // OK
```

### "Signal not found in scope"

```rust
// Problem: signal not in closure scope
let signal = use_signal(|| 0);
spawn(async {
    signal += 1;  // Error: signal not captured
});

// Solution: move into closure
spawn(async move {
    signal += 1;  // OK
});
```

### "Component re-renders too much"

```rust
// Problem: reading signal in component body
fn Component() -> Element {
    let data = expensive_signal.read();  // Re-renders on every change

    rsx! { "{data}" }
}

// Solution: use memos
fn Component() -> Element {
    let processed = use_memo(move || {
        process(expensive_signal())
    });

    rsx! { "{processed}" }
}
```

---

## Resources

- [Official Docs](https://dioxuslabs.com/learn/0.6/guide)
- [API Reference](https://docs.rs/dioxus/latest/dioxus/)
- [Examples](https://github.com/DioxusLabs/dioxus/tree/main/examples)
- [Discord Community](https://discord.gg/XgGxMSkvUM)
