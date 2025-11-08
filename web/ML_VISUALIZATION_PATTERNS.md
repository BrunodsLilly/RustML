# ML Visualization Patterns in Dioxus 0.6

This document provides concrete implementation patterns for common machine learning visualization scenarios in Dioxus.

## Table of Contents

1. [Linear Regression Visualization](#linear-regression-visualization)
2. [Gradient Descent Animation](#gradient-descent-animation)
3. [Neural Network Architecture Diagram](#neural-network-architecture-diagram)
4. [Loss Curve Real-Time Plot](#loss-curve-real-time-plot)
5. [Confusion Matrix Heatmap](#confusion-matrix-heatmap)
6. [Decision Boundary Visualization](#decision-boundary-visualization)
7. [Feature Space Explorer](#feature-space-explorer)
8. [Model Comparison Dashboard](#model-comparison-dashboard)

---

## Linear Regression Visualization

Interactive visualization showing data points, regression line, and residuals.

```rust
use dioxus::prelude::*;
use linear_algebra::Vector;
use linear_regression::LinearRegression;

#[derive(Clone, PartialEq)]
struct DataPoint {
    x: f64,
    y: f64,
}

fn LinearRegressionDemo() -> Element {
    let mut data = use_signal(|| vec![
        DataPoint { x: 1.0, y: 2.0 },
        DataPoint { x: 2.0, y: 4.0 },
        DataPoint { x: 3.0, y: 5.0 },
        DataPoint { x: 4.0, y: 4.0 },
        DataPoint { x: 5.0, y: 5.0 },
    ]);

    let model = use_memo(move || {
        let x_values: Vec<f64> = data().iter().map(|p| p.x).collect();
        let y_values: Vec<f64> = data().iter().map(|p| p.y).collect();

        let mut model = LinearRegression::new();
        model.fit(&Vector::from(x_values), &Vector::from(y_values));
        model
    });

    let mut show_residuals = use_signal(|| false);
    let mut adding_point = use_signal(|| false);

    let add_point = move |evt: MouseEvent| {
        if adding_point() {
            let x = (evt.page_coordinates().x - 50.0) / 80.0;
            let y = (350.0 - evt.page_coordinates().y) / 50.0;
            data.write().push(DataPoint { x, y });
        }
    };

    rsx! {
        div { class: "linear-regression-demo",
            h2 { "Linear Regression Interactive Demo" }

            div { class: "controls",
                button {
                    onclick: move |_| adding_point.toggle(),
                    class: if adding_point() { "active" } else { "" },
                    if adding_point() { "Click plot to add point" } else { "Add points" }
                }

                button {
                    onclick: move |_| show_residuals.toggle(),
                    if show_residuals() { "Hide residuals" } else { "Show residuals" }
                }

                button {
                    onclick: move |_| data.set(vec![]),
                    "Clear"
                }
            }

            svg {
                width: "600",
                height: "400",
                view_box: "0 0 600 400",
                onclick: add_point,

                // Grid
                g { class: "grid",
                    for i in 0..=10 {
                        line {
                            x1: "50", x2: "550",
                            y1: "{50 + i * 30}",
                            y2: "{50 + i * 30}",
                            stroke: "#e0e0e0",
                            stroke_width: "1"
                        }
                        line {
                            x1: "{50 + i * 50}",
                            x2: "{50 + i * 50}",
                            y1: "50", y2: "350",
                            stroke: "#e0e0e0",
                            stroke_width: "1"
                        }
                    }
                }

                // Axes
                g { class: "axes",
                    line {
                        x1: "50", y1: "350",
                        x2: "550", y2: "350",
                        stroke: "black",
                        stroke_width: "2"
                    }
                    line {
                        x1: "50", y1: "50",
                        x2: "50", y2: "350",
                        stroke: "black",
                        stroke_width: "2"
                    }

                    // Axis labels
                    text { x: "300", y: "385", text_anchor: "middle", "X" }
                    text { x: "25", y: "200", text_anchor: "middle", "Y" }
                }

                // Regression line
                if !data().is_empty() {
                    {
                        let m = model();
                        let slope = m.slope();
                        let intercept = m.intercept();

                        let x1 = 0.0;
                        let y1 = slope * x1 + intercept;
                        let x2 = 6.0;
                        let y2 = slope * x2 + intercept;

                        rsx! {
                            line {
                                x1: "{50.0 + x1 * 80.0}",
                                y1: "{350.0 - y1 * 50.0}",
                                x2: "{50.0 + x2 * 80.0}",
                                y2: "{350.0 - y2 * 50.0}",
                                stroke: "red",
                                stroke_width: "2"
                            }
                        }
                    }
                }

                // Residuals
                if show_residuals() {
                    g { class: "residuals",
                        for point in data().iter() {
                            {
                                let predicted_y = model().predict(point.x);
                                let x_pos = 50.0 + point.x * 80.0;

                                rsx! {
                                    line {
                                        x1: "{x_pos}",
                                        y1: "{350.0 - point.y * 50.0}",
                                        x2: "{x_pos}",
                                        y2: "{350.0 - predicted_y * 50.0}",
                                        stroke: "orange",
                                        stroke_width: "1",
                                        stroke_dasharray: "5,5"
                                    }
                                }
                            }
                        }
                    }
                }

                // Data points
                g { class: "data-points",
                    for point in data().iter() {
                        circle {
                            cx: "{50.0 + point.x * 80.0}",
                            cy: "{350.0 - point.y * 50.0}",
                            r: "5",
                            fill: "blue",
                            stroke: "darkblue",
                            stroke_width: "2"
                        }
                    }
                }
            }

            // Statistics panel
            if !data().is_empty() {
                div { class: "statistics",
                    h3 { "Model Statistics" }
                    p { "Slope: {model().slope():.4}" }
                    p { "Intercept: {model().intercept():.4}" }
                    p { "R²: {model().r_squared():.4}" }
                    p { "Data points: {data().len()}" }
                }
            }
        }
    }
}
```

---

## Gradient Descent Animation

Animated visualization of gradient descent optimization.

```rust
use dioxus::prelude::*;
use std::time::Duration;

#[derive(Clone, Copy)]
struct Point2D {
    x: f64,
    y: f64,
}

fn GradientDescentAnimation() -> Element {
    let mut position = use_signal(|| Point2D { x: -2.0, y: 3.0 });
    let mut path = use_signal(|| vec![Point2D { x: -2.0, y: 3.0 }]);
    let mut is_running = use_signal(|| false);
    let mut learning_rate = use_signal(|| 0.1);

    // Loss function: f(x, y) = x² + y²
    let loss = |p: Point2D| p.x * p.x + p.y * p.y;

    // Gradient: (2x, 2y)
    let gradient = |p: Point2D| Point2D { x: 2.0 * p.x, y: 2.0 * p.y };

    // Animation coroutine
    use_coroutine(move |_: UnboundedReceiver<()>| async move {
        loop {
            if is_running() {
                let current = position();
                let grad = gradient(current);

                // Gradient descent step
                let new_pos = Point2D {
                    x: current.x - learning_rate() * grad.x,
                    y: current.y - learning_rate() * grad.y,
                };

                position.set(new_pos);
                path.write().push(new_pos);

                // Stop if converged
                if loss(new_pos).abs() < 0.01 {
                    is_running.set(false);
                }
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    });

    rsx! {
        div { class: "gradient-descent-demo",
            h2 { "Gradient Descent Visualization" }

            div { class: "controls",
                button {
                    onclick: move |_| is_running.toggle(),
                    if is_running() { "Pause" } else { "Start" }
                }

                button {
                    onclick: move |_| {
                        position.set(Point2D { x: -2.0, y: 3.0 });
                        path.set(vec![Point2D { x: -2.0, y: 3.0 }]);
                    },
                    "Reset"
                }

                label {
                    "Learning rate: "
                    input {
                        r#type: "range",
                        min: "0.01",
                        max: "0.5",
                        step: "0.01",
                        value: "{learning_rate()}",
                        oninput: move |evt| {
                            if let Ok(val) = evt.value().parse::<f64>() {
                                learning_rate.set(val);
                            }
                        }
                    }
                    span { "{learning_rate():.2}" }
                }
            }

            svg {
                width: "600",
                height: "600",
                view_box: "-4 -4 8 8",

                // Contour plot of loss function
                g { class: "contours",
                    for level in [0.5, 1.0, 2.0, 4.0, 8.0] {
                        circle {
                            cx: "0",
                            cy: "0",
                            r: "{level.sqrt()}",
                            fill: "none",
                            stroke: "#ccc",
                            stroke_width: "0.05"
                        }
                    }
                }

                // Axes
                line { x1: "-4", y1: "0", x2: "4", y2: "0", stroke: "black", stroke_width: "0.02" }
                line { x1: "0", y1: "-4", x2: "0", y2: "4", stroke: "black", stroke_width: "0.02" }

                // Path taken by gradient descent
                polyline {
                    points: {
                        path()
                            .iter()
                            .map(|p| format!("{},{}", p.x, p.y))
                            .collect::<Vec<_>>()
                            .join(" ")
                    },
                    fill: "none",
                    stroke: "blue",
                    stroke_width: "0.05",
                    opacity: "0.6"
                }

                // Path points
                for (i, point) in path().iter().enumerate() {
                    circle {
                        cx: "{point.x}",
                        cy: "{point.y}",
                        r: "0.1",
                        fill: "blue",
                        opacity: "{0.3 + 0.7 * (i as f64 / path().len() as f64)}"
                    }
                }

                // Current position
                circle {
                    cx: "{position().x}",
                    cy: "{position().y}",
                    r: "0.15",
                    fill: "red",
                    stroke: "darkred",
                    stroke_width: "0.03"
                }

                // Gradient arrow
                {
                    let grad = gradient(position());
                    let arrow_scale = 0.3;

                    rsx! {
                        line {
                            x1: "{position().x}",
                            y1: "{position().y}",
                            x2: "{position().x - grad.x * arrow_scale}",
                            y2: "{position().y - grad.y * arrow_scale}",
                            stroke: "orange",
                            stroke_width: "0.05",
                            marker_end: "url(#arrowhead)"
                        }
                    }
                }

                // Arrow marker definition
                defs {
                    marker {
                        id: "arrowhead",
                        marker_width: "10",
                        marker_height: "10",
                        refX: "5",
                        refY: "5",
                        orient: "auto",
                        polygon {
                            points: "0 0, 10 5, 0 10",
                            fill: "orange"
                        }
                    }
                }
            }

            div { class: "info",
                p { "Position: ({position().x:.4}, {position().y:.4})" }
                p { "Loss: {loss(position()):.6}" }
                p { "Steps: {path().len()}" }
            }
        }
    }
}
```

---

## Neural Network Architecture Diagram

Interactive diagram showing neural network layers and connections.

```rust
use dioxus::prelude::*;

#[derive(Clone, PartialEq)]
struct Layer {
    neurons: usize,
    activation: String,
}

fn NeuralNetworkDiagram() -> Element {
    let layers = use_signal(|| vec![
        Layer { neurons: 3, activation: "Input".into() },
        Layer { neurons: 4, activation: "ReLU".into() },
        Layer { neurons: 4, activation: "ReLU".into() },
        Layer { neurons: 2, activation: "Softmax".into() },
    ]);

    let mut highlighted_layer = use_signal(|| None::<usize>);

    rsx! {
        div { class: "neural-network-diagram",
            h2 { "Neural Network Architecture" }

            svg {
                width: "800",
                height: "400",
                view_box: "0 0 800 400",

                // Draw layers
                for (layer_idx, layer) in layers().iter().enumerate() {
                    {
                        let x = 100.0 + layer_idx as f64 * 200.0;
                        let layer_height = layer.neurons as f64 * 60.0;
                        let start_y = (400.0 - layer_height) / 2.0;

                        rsx! {
                            g {
                                class: "layer",
                                onmouseenter: move |_| highlighted_layer.set(Some(layer_idx)),
                                onmouseleave: move |_| highlighted_layer.set(None),

                                // Draw connections to next layer
                                if layer_idx < layers().len() - 1 {
                                    {
                                        let next_layer = &layers()[layer_idx + 1];
                                        let next_x = x + 200.0;
                                        let next_layer_height = next_layer.neurons as f64 * 60.0;
                                        let next_start_y = (400.0 - next_layer_height) / 2.0;

                                        rsx! {
                                            g { class: "connections",
                                                for i in 0..layer.neurons {
                                                    for j in 0..next_layer.neurons {
                                                        line {
                                                            x1: "{x + 20.0}",
                                                            y1: "{start_y + i as f64 * 60.0 + 30.0}",
                                                            x2: "{next_x - 20.0}",
                                                            y2: "{next_start_y + j as f64 * 60.0 + 30.0}",
                                                            stroke: if highlighted_layer() == Some(layer_idx) {
                                                                "#4CAF50"
                                                            } else {
                                                                "#ccc"
                                                            },
                                                            stroke_width: "1",
                                                            opacity: "0.3"
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                // Draw neurons
                                for neuron_idx in 0..layer.neurons {
                                    circle {
                                        cx: "{x}",
                                        cy: "{start_y + neuron_idx as f64 * 60.0 + 30.0}",
                                        r: "20",
                                        fill: if highlighted_layer() == Some(layer_idx) {
                                            "#4CAF50"
                                        } else {
                                            "#2196F3"
                                        },
                                        stroke: "#1976D2",
                                        stroke_width: "2"
                                    }
                                }

                                // Layer label
                                text {
                                    x: "{x}",
                                    y: "{start_y - 20.0}",
                                    text_anchor: "middle",
                                    font_weight: "bold",
                                    "{layer.activation}"
                                }

                                text {
                                    x: "{x}",
                                    y: "{start_y + layer_height + 30.0}",
                                    text_anchor: "middle",
                                    font_size: "12",
                                    "{layer.neurons} neurons"
                                }
                            }
                        }
                    }
                }
            }

            div { class: "controls",
                h3 { "Add Layer" }
                button {
                    onclick: move |_| {
                        let mut l = layers.write();
                        l.insert(l.len() - 1, Layer {
                            neurons: 4,
                            activation: "ReLU".into()
                        });
                    },
                    "Add Hidden Layer"
                }
            }

            if let Some(idx) = highlighted_layer() {
                div { class: "layer-info",
                    h3 { "Layer {idx}" }
                    p { "Neurons: {layers()[idx].neurons}" }
                    p { "Activation: {layers()[idx].activation}" }
                }
            }
        }
    }
}
```

---

## Loss Curve Real-Time Plot

Real-time plotting of training/validation loss.

```rust
use dioxus::prelude::*;
use std::time::Duration;

#[derive(Clone, Copy)]
struct LossPoint {
    epoch: usize,
    train_loss: f64,
    val_loss: f64,
}

fn LossCurvePlot() -> Element {
    let mut loss_history = use_signal(|| Vec::<LossPoint>::new());
    let mut is_training = use_signal(|| false);
    let mut current_epoch = use_signal(|| 0);

    // Simulated training coroutine
    use_coroutine(move |_: UnboundedReceiver<()>| async move {
        loop {
            if is_training() {
                let epoch = current_epoch();

                // Simulate decreasing loss with noise
                let train_loss = 2.0 * (-epoch as f64 * 0.1).exp()
                    + (epoch as f64 * 0.5).sin() * 0.1;
                let val_loss = 2.0 * (-epoch as f64 * 0.1).exp()
                    + (epoch as f64 * 0.5).cos() * 0.15
                    + 0.1;

                loss_history.write().push(LossPoint {
                    epoch,
                    train_loss,
                    val_loss,
                });

                current_epoch += 1;

                if epoch >= 100 {
                    is_training.set(false);
                }
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    });

    rsx! {
        div { class: "loss-curve-plot",
            h2 { "Training Progress" }

            div { class: "controls",
                button {
                    onclick: move |_| is_training.toggle(),
                    disabled: current_epoch() >= 100,
                    if is_training() { "Pause" } else { "Start Training" }
                }

                button {
                    onclick: move |_| {
                        loss_history.set(vec![]);
                        current_epoch.set(0);
                    },
                    "Reset"
                }
            }

            svg {
                width: "700",
                height: "400",
                view_box: "0 0 700 400",

                // Grid
                g { class: "grid",
                    for i in 0..=10 {
                        line {
                            x1: "60", x2: "660",
                            y1: "{50 + i * 30}",
                            y2: "{50 + i * 30}",
                            stroke: "#f0f0f0"
                        }
                    }
                    for i in 0..=10 {
                        line {
                            x1: "{60 + i * 60}",
                            x2: "{60 + i * 60}",
                            y1: "50", y2: "350",
                            stroke: "#f0f0f0"
                        }
                    }
                }

                // Axes
                line { x1: "60", y1: "350", x2: "660", y2: "350", stroke: "black", stroke_width: "2" }
                line { x1: "60", y1: "50", x2: "60", y2: "350", stroke: "black", stroke_width: "2" }

                // Axis labels
                text { x: "360", y: "385", text_anchor: "middle", "Epoch" }
                text {
                    x: "30", y: "200",
                    text_anchor: "middle",
                    transform: "rotate(-90, 30, 200)",
                    "Loss"
                }

                if !loss_history().is_empty() {
                    {
                        let max_loss = loss_history()
                            .iter()
                            .map(|p| p.train_loss.max(p.val_loss))
                            .fold(0.0, f64::max);

                        let scale_x = |epoch: usize| 60.0 + (epoch as f64 * 6.0);
                        let scale_y = |loss: f64| 350.0 - (loss / max_loss * 280.0);

                        rsx! {
                            // Training loss line
                            polyline {
                                points: {
                                    loss_history()
                                        .iter()
                                        .map(|p| format!("{},{}", scale_x(p.epoch), scale_y(p.train_loss)))
                                        .collect::<Vec<_>>()
                                        .join(" ")
                                },
                                fill: "none",
                                stroke: "#2196F3",
                                stroke_width: "2"
                            }

                            // Validation loss line
                            polyline {
                                points: {
                                    loss_history()
                                        .iter()
                                        .map(|p| format!("{},{}", scale_x(p.epoch), scale_y(p.val_loss)))
                                        .collect::<Vec<_>>()
                                        .join(" ")
                                },
                                fill: "none",
                                stroke: "#FF9800",
                                stroke_width: "2",
                                stroke_dasharray: "5,5"
                            }
                        }
                    }
                }

                // Legend
                g { class: "legend",
                    rect { x: "500", y: "70", width: "140", height: "60", fill: "white", stroke: "black" }
                    line { x1: "510", y1: "90", x2: "540", y2: "90", stroke: "#2196F3", stroke_width: "2" }
                    text { x: "550", y: "95", "Training Loss" }
                    line {
                        x1: "510", y1: "110", x2: "540", y2: "110",
                        stroke: "#FF9800",
                        stroke_width: "2",
                        stroke_dasharray: "5,5"
                    }
                    text { x: "550", y: "115", "Validation Loss" }
                }
            }

            div { class: "metrics",
                if let Some(latest) = loss_history().last() {
                    p { "Epoch: {latest.epoch}" }
                    p { "Train Loss: {latest.train_loss:.4}" }
                    p { "Val Loss: {latest.val_loss:.4}" }
                }
            }
        }
    }
}
```

---

## Confusion Matrix Heatmap

Interactive confusion matrix for classification models.

```rust
use dioxus::prelude::*;

type ConfusionMatrix = Vec<Vec<usize>>;

fn ConfusionMatrixHeatmap() -> Element {
    let matrix = use_signal(|| vec![
        vec![50, 2, 0, 1],
        vec![1, 45, 3, 1],
        vec![0, 2, 48, 0],
        vec![2, 1, 0, 47],
    ]);

    let class_names = vec!["Cat", "Dog", "Bird", "Fish"];
    let mut hovered_cell = use_signal(|| None::<(usize, usize)>);

    let max_value = matrix()
        .iter()
        .flat_map(|row| row.iter())
        .max()
        .copied()
        .unwrap_or(1);

    let get_color = |value: usize| {
        let intensity = (value as f64 / max_value as f64 * 255.0) as u8;
        format!("rgb({}, {}, {})", 255 - intensity, 255 - intensity, 255)
    };

    rsx! {
        div { class: "confusion-matrix",
            h2 { "Confusion Matrix" }

            svg {
                width: "500",
                height: "500",
                view_box: "0 0 500 500",

                // Column labels (Predicted)
                text {
                    x: "250", y: "30",
                    text_anchor: "middle",
                    font_weight: "bold",
                    "Predicted Class"
                }

                for (i, name) in class_names.iter().enumerate() {
                    text {
                        x: "{100 + i * 80 + 40}",
                        y: "60",
                        text_anchor: "middle",
                        "{name}"
                    }
                }

                // Row labels (Actual)
                text {
                    x: "30", y: "250",
                    text_anchor: "middle",
                    font_weight: "bold",
                    transform: "rotate(-90, 30, 250)",
                    "Actual Class"
                }

                for (i, name) in class_names.iter().enumerate() {
                    text {
                        x: "80",
                        y: "{80 + i * 80 + 45}",
                        text_anchor: "end",
                        "{name}"
                    }
                }

                // Matrix cells
                for (i, row) in matrix().iter().enumerate() {
                    for (j, &value) in row.iter().enumerate() {
                        g {
                            onmouseenter: move |_| hovered_cell.set(Some((i, j))),
                            onmouseleave: move |_| hovered_cell.set(None),

                            rect {
                                x: "{100 + j * 80}",
                                y: "{80 + i * 80}",
                                width: "75",
                                height: "75",
                                fill: get_color(value),
                                stroke: if hovered_cell() == Some((i, j)) {
                                    "red"
                                } else {
                                    "black"
                                },
                                stroke_width: if hovered_cell() == Some((i, j)) {
                                    "3"
                                } else {
                                    "1"
                                }
                            }

                            text {
                                x: "{100 + j * 80 + 37.5}",
                                y: "{80 + i * 80 + 45}",
                                text_anchor: "middle",
                                font_size: "20",
                                fill: if value > max_value / 2 { "white" } else { "black" },
                                "{value}"
                            }
                        }
                    }
                }
            }

            if let Some((i, j)) = hovered_cell() {
                div { class: "cell-info",
                    p { "Actual: {class_names[i]}" }
                    p { "Predicted: {class_names[j]}" }
                    p { "Count: {matrix()[i][j]}" }
                    p {
                        if i == j {
                            "Correct prediction"
                        } else {
                            "Misclassification"
                        }
                    }
                }
            }

            // Metrics
            div { class: "metrics",
                h3 { "Overall Metrics" }
                {
                    let total: usize = matrix().iter().flat_map(|row| row.iter()).sum();
                    let correct: usize = matrix()
                        .iter()
                        .enumerate()
                        .map(|(i, row)| row[i])
                        .sum();
                    let accuracy = correct as f64 / total as f64;

                    rsx! {
                        p { "Total Predictions: {total}" }
                        p { "Correct: {correct}" }
                        p { "Accuracy: {accuracy * 100.0:.2}%" }
                    }
                }
            }
        }
    }
}
```

---

## Feature Space Explorer

Interactive 2D feature space visualization with decision boundaries.

```rust
use dioxus::prelude::*;

#[derive(Clone, Copy, PartialEq)]
struct Point {
    x: f64,
    y: f64,
    class: usize,
}

fn FeatureSpaceExplorer() -> Element {
    let mut data = use_signal(|| Vec::<Point>::new());
    let mut current_class = use_signal(|| 0);
    let class_colors = vec!["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"];

    let add_point = move |evt: MouseEvent| {
        let svg_rect = evt.element().get_bounding_client_rect();
        let x = (evt.page_coordinates().x - svg_rect.x() - 50.0) / 5.0;
        let y = (350.0 - (evt.page_coordinates().y - svg_rect.y())) / 5.0;

        if x >= 0.0 && x <= 100.0 && y >= 0.0 && y <= 60.0 {
            data.write().push(Point {
                x,
                y,
                class: current_class(),
            });
        }
    };

    rsx! {
        div { class: "feature-space-explorer",
            h2 { "Feature Space Explorer" }

            div { class: "controls",
                p { "Click on plot to add points" }

                div { class: "class-selector",
                    for (i, color) in class_colors.iter().enumerate() {
                        button {
                            onclick: move |_| current_class.set(i),
                            class: if current_class() == i { "active" } else { "" },
                            style: "background-color: {color};",
                            "Class {i}"
                        }
                    }
                }

                button {
                    onclick: move |_| data.set(vec![]),
                    "Clear All"
                }
            }

            svg {
                width: "600",
                height: "400",
                view_box: "0 0 600 400",
                onclick: add_point,

                // Background
                rect {
                    x: "50", y: "50",
                    width: "500", height: "300",
                    fill: "#f9f9f9",
                    stroke: "none"
                }

                // Grid
                g { class: "grid",
                    for i in 0..=10 {
                        line {
                            x1: "50", x2: "550",
                            y1: "{50 + i * 30}",
                            y2: "{50 + i * 30}",
                            stroke: "#e0e0e0"
                        }
                        line {
                            x1: "{50 + i * 50}",
                            x2: "{50 + i * 50}",
                            y1: "50", y2: "350",
                            stroke: "#e0e0e0"
                        }
                    }
                }

                // Axes
                line { x1: "50", y1: "350", x2: "550", y2: "350", stroke: "black", stroke_width: "2" }
                line { x1: "50", y1: "50", x2: "50", y2: "350", stroke: "black", stroke_width: "2" }

                text { x: "300", y: "385", text_anchor: "middle", "Feature 1" }
                text {
                    x: "25", y: "200",
                    text_anchor: "middle",
                    transform: "rotate(-90, 25, 200)",
                    "Feature 2"
                }

                // Data points
                for point in data().iter() {
                    circle {
                        cx: "{50.0 + point.x * 5.0}",
                        cy: "{350.0 - point.y * 5.0}",
                        r: "6",
                        fill: class_colors[point.class],
                        stroke: "white",
                        stroke_width: "2",
                        opacity: "0.8"
                    }
                }

                // Current class indicator
                circle {
                    cx: "575",
                    cy: "75",
                    r: "8",
                    fill: class_colors[current_class()]
                }
            }

            div { class: "statistics",
                h3 { "Dataset Statistics" }
                for (i, color) in class_colors.iter().enumerate() {
                    {
                        let count = data().iter().filter(|p| p.class == i).count();
                        rsx! {
                            p {
                                span {
                                    style: "color: {color}; font-weight: bold;",
                                    "Class {i}: "
                                }
                                "{count} points"
                            }
                        }
                    }
                }
                p { "Total: {data().len()} points" }
            }
        }
    }
}
```

---

## Responsive Chart Component

Reusable responsive chart that adapts to container size.

```rust
use dioxus::prelude::*;
use wasm_bindgen::prelude::*;
use web_sys::window;

#[component]
fn ResponsiveChart(
    data: Vec<f64>,
    title: String,
    x_label: String,
    y_label: String,
) -> Element {
    let mut dimensions = use_signal(|| (800, 400));
    let container_ref = use_signal(|| None::<web_sys::Element>);

    // Resize observer
    use_effect(move || {
        if let Some(container) = container_ref() {
            let closure = Closure::wrap(Box::new(move |_entries: js_sys::Array| {
                if let Some(element) = container_ref() {
                    let width = element.client_width();
                    let height = element.client_height();
                    dimensions.set((width as i32, height as i32));
                }
            }) as Box<dyn FnMut(_)>);

            let observer = web_sys::ResizeObserver::new(closure.as_ref().unchecked_ref()).unwrap();
            observer.observe(&container);

            closure.forget();
        }
    });

    let (width, height) = dimensions();
    let margin = 60;
    let plot_width = width - 2 * margin;
    let plot_height = height - 2 * margin;

    let max_value = data.iter().copied().fold(f64::MIN, f64::max);
    let scale_y = |val: f64| margin + plot_height - ((val / max_value) * plot_height as f64) as i32;
    let scale_x = |idx: usize| margin + (idx * plot_width as usize / data.len().max(1)) as i32;

    rsx! {
        div {
            class: "responsive-chart-container",
            style: "width: 100%; height: 100%;",
            onmounted: move |evt| {
                container_ref.set(Some(evt.data.downcast::<web_sys::Element>()));
            },

            svg {
                width: "{width}",
                height: "{height}",
                view_box: "0 0 {width} {height}",

                // Title
                text {
                    x: "{width / 2}",
                    y: "30",
                    text_anchor: "middle",
                    font_size: "18",
                    font_weight: "bold",
                    "{title}"
                }

                // Axes
                line {
                    x1: "{margin}",
                    y1: "{height - margin}",
                    x2: "{width - margin}",
                    y2: "{height - margin}",
                    stroke: "black",
                    stroke_width: "2"
                }
                line {
                    x1: "{margin}",
                    y1: "{margin}",
                    x2: "{margin}",
                    y2: "{height - margin}",
                    stroke: "black",
                    stroke_width: "2"
                }

                // Axis labels
                text {
                    x: "{width / 2}",
                    y: "{height - 10}",
                    text_anchor: "middle",
                    "{x_label}"
                }
                text {
                    x: "20",
                    y: "{height / 2}",
                    text_anchor: "middle",
                    transform: "rotate(-90, 20, {height / 2})",
                    "{y_label}"
                }

                // Data line
                polyline {
                    points: {
                        data
                            .iter()
                            .enumerate()
                            .map(|(i, &val)| format!("{},{}", scale_x(i), scale_y(val)))
                            .collect::<Vec<_>>()
                            .join(" ")
                    },
                    fill: "none",
                    stroke: "#2196F3",
                    stroke_width: "2"
                }

                // Data points
                for (i, &val) in data.iter().enumerate() {
                    circle {
                        cx: "{scale_x(i)}",
                        cy: "{scale_y(val)}",
                        r: "4",
                        fill: "#2196F3"
                    }
                }
            }
        }
    }
}
```

---

## Summary

These patterns demonstrate:

1. **Interactive Visualizations**: Click-to-add points, hover effects, drag interactions
2. **Real-Time Updates**: Animated training progress, live loss curves
3. **Modular Components**: Reusable chart components with props
4. **Responsive Design**: Charts that adapt to container size
5. **State Management**: Signals, memos, and coroutines for reactive data flow
6. **SVG Rendering**: All visualizations use native SVG for crisp, scalable graphics

All examples can be combined and customized for specific ML educational use cases. The patterns shown here provide a foundation for building sophisticated, interactive ML visualizations in Dioxus 0.6.
