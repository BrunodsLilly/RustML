use crate::components::*;
use crate::routes::Route;
use dioxus::prelude::*;
use linear_algebra::vectors::Vector;
use linear_algebra::{matrix::Matrix, v};
use linear_regression::LinearRegressor;

#[component]
pub fn ShowcaseView() -> Element {
    rsx! {
        div { id: "showcase",
            NavBar {}
            div { class: "container",
                h1 { "ML Library Showcase" }
                p { "Interactive demonstrations of the machine learning libraries built in Rust" }

                VectorDemo {}
                VectorOperationsDemo {}
                MatrixOperationsDemo {}
                GradientDescentDemo {}

                div { class: "navigation",
                    Link {
                        to: Route::MainView {},
                        "← Back to Main"
                    }
                }
            }
        }
    }
}

#[component]
fn VectorDemo() -> Element {
    rsx! {
        section { class: "demo-section",
            h2 { "Vector Operations" }
            p { "Demonstrating basic vector arithmetic operations" }

            div { class: "demo-box",
                h3 { "Vector Addition" }
                code { "v![1, 2, 3] + v![4, 5, 6]" }
                pre {
                    {format!("Result: {:?}", (v![1, 2, 3] + v![4, 5, 6]).data)}
                }
            }

            div { class: "demo-box",
                h3 { "Vector Scalar Multiplication" }
                code { "v![1, 2, 3] * 10" }
                pre {
                    {format!("Result: {:?}", (v![1, 2, 3] * 10).data)}
                }
            }

            div { class: "demo-box",
                h3 { "Dot Product" }
                code { "v![1, 2, 3].dot(&v![4, 5, 6])" }
                pre {
                    {format!("Result: {}", v![1, 2, 3].dot(&v![4, 5, 6]))}
                }
            }
        }
    }
}

#[component]
fn VectorOperationsDemo() -> Element {
    let mut vector_a = use_signal(|| vec![1.0, 2.0, 3.0]);
    let mut vector_b = use_signal(|| vec![4.0, 5.0, 6.0]);
    let mut scalar = use_signal(|| 2.0);

    let result_add = use_memo(move || {
        let va = Vector { data: vector_a() };
        let vb = Vector { data: vector_b() };
        (va + vb).data
    });

    let result_dot = use_memo(move || {
        let va = Vector { data: vector_a() };
        let vb = Vector { data: vector_b() };
        va.dot(&vb)
    });

    let result_scalar = use_memo(move || {
        let va = Vector { data: vector_a() };
        (va * scalar()).data
    });

    rsx! {
        section { class: "demo-section interactive",
            h2 { "Interactive Vector Calculator" }
            p { "Modify the vectors and see operations in real-time" }

            div { class: "inputs",
                div { class: "input-group",
                    label { "Vector A:" }
                    div { class: "vector-inputs",
                        input {
                            r#type: "number",
                            value: "{vector_a()[0]}",
                            step: "0.1",
                            oninput: move |evt| {
                                if let Ok(val) = evt.value().parse::<f64>() {
                                    vector_a.write()[0] = val;
                                }
                            }
                        }
                        input {
                            r#type: "number",
                            value: "{vector_a()[1]}",
                            step: "0.1",
                            oninput: move |evt| {
                                if let Ok(val) = evt.value().parse::<f64>() {
                                    vector_a.write()[1] = val;
                                }
                            }
                        }
                        input {
                            r#type: "number",
                            value: "{vector_a()[2]}",
                            step: "0.1",
                            oninput: move |evt| {
                                if let Ok(val) = evt.value().parse::<f64>() {
                                    vector_a.write()[2] = val;
                                }
                            }
                        }
                    }
                }

                div { class: "input-group",
                    label { "Vector B:" }
                    div { class: "vector-inputs",
                        input {
                            r#type: "number",
                            value: "{vector_b()[0]}",
                            step: "0.1",
                            oninput: move |evt| {
                                if let Ok(val) = evt.value().parse::<f64>() {
                                    vector_b.write()[0] = val;
                                }
                            }
                        }
                        input {
                            r#type: "number",
                            value: "{vector_b()[1]}",
                            step: "0.1",
                            oninput: move |evt| {
                                if let Ok(val) = evt.value().parse::<f64>() {
                                    vector_b.write()[1] = val;
                                }
                            }
                        }
                        input {
                            r#type: "number",
                            value: "{vector_b()[2]}",
                            step: "0.1",
                            oninput: move |evt| {
                                if let Ok(val) = evt.value().parse::<f64>() {
                                    vector_b.write()[2] = val;
                                }
                            }
                        }
                    }
                }

                div { class: "input-group",
                    label { "Scalar:" }
                    input {
                        r#type: "number",
                        value: "{scalar()}",
                        step: "0.1",
                        oninput: move |evt| {
                            if let Ok(val) = evt.value().parse::<f64>() {
                                scalar.set(val);
                            }
                        }
                    }
                }
            }

            div { class: "results",
                h3 { "Results:" }
                div { class: "result-box",
                    strong { "A + B = " }
                    code { "{result_add:?}" }
                }
                div { class: "result-box",
                    strong { "A · B = " }
                    code { "{result_dot()}" }
                }
                div { class: "result-box",
                    strong { "A × scalar = " }
                    code { "{result_scalar:?}" }
                }
            }
        }
    }
}

#[component]
fn MatrixOperationsDemo() -> Element {
    // Matrix A state (2x2)
    let mut a11 = use_signal(|| 1.0);
    let mut a12 = use_signal(|| 2.0);
    let mut a21 = use_signal(|| 3.0);
    let mut a22 = use_signal(|| 4.0);

    // Matrix B state (2x2)
    let mut b11 = use_signal(|| 5.0);
    let mut b12 = use_signal(|| 6.0);
    let mut b21 = use_signal(|| 7.0);
    let mut b22 = use_signal(|| 8.0);

    // Vector v state (2x1)
    let mut v1 = use_signal(|| 1.0);
    let mut v2 = use_signal(|| 2.0);

    // Compute results
    let matrix_a = use_memo(move || {
        Matrix::from_vec(vec![a11(), a12(), a21(), a22()], 2, 2).unwrap()
    });

    let matrix_b = use_memo(move || {
        Matrix::from_vec(vec![b11(), b12(), b21(), b22()], 2, 2).unwrap()
    });

    let vector_v = use_memo(move || {
        Vector { data: vec![v1(), v2()] }
    });

    let result_add = use_memo(move || {
        matrix_a() + matrix_b()
    });

    let result_mul = use_memo(move || {
        matrix_a() * matrix_b()
    });

    let result_transpose = use_memo(move || {
        matrix_a().transpose()
    });

    let result_mat_vec = use_memo(move || {
        matrix_a() * vector_v()
    });

    rsx! {
        section { class: "demo-section matrix-ops",
            h2 { "🔢 Interactive Matrix Operations" }
            p { "Explore matrix arithmetic with our linear algebra library" }

            div { class: "matrix-container",
                // Left panel: Input matrices
                div { class: "matrix-panel",
                    h3 { "Input Matrices" }

                    div { class: "matrix-input-group",
                        h4 { "Matrix A (2×2)" }
                        div { class: "matrix-grid",
                            input {
                                r#type: "number",
                                step: "0.1",
                                value: "{a11()}",
                                oninput: move |evt| {
                                    if let Ok(val) = evt.value().parse::<f64>() {
                                        a11.set(val);
                                    }
                                }
                            }
                            input {
                                r#type: "number",
                                step: "0.1",
                                value: "{a12()}",
                                oninput: move |evt| {
                                    if let Ok(val) = evt.value().parse::<f64>() {
                                        a12.set(val);
                                    }
                                }
                            }
                            input {
                                r#type: "number",
                                step: "0.1",
                                value: "{a21()}",
                                oninput: move |evt| {
                                    if let Ok(val) = evt.value().parse::<f64>() {
                                        a21.set(val);
                                    }
                                }
                            }
                            input {
                                r#type: "number",
                                step: "0.1",
                                value: "{a22()}",
                                oninput: move |evt| {
                                    if let Ok(val) = evt.value().parse::<f64>() {
                                        a22.set(val);
                                    }
                                }
                            }
                        }
                    }

                    div { class: "matrix-input-group",
                        h4 { "Matrix B (2×2)" }
                        div { class: "matrix-grid",
                            input {
                                r#type: "number",
                                step: "0.1",
                                value: "{b11()}",
                                oninput: move |evt| {
                                    if let Ok(val) = evt.value().parse::<f64>() {
                                        b11.set(val);
                                    }
                                }
                            }
                            input {
                                r#type: "number",
                                step: "0.1",
                                value: "{b12()}",
                                oninput: move |evt| {
                                    if let Ok(val) = evt.value().parse::<f64>() {
                                        b12.set(val);
                                    }
                                }
                            }
                            input {
                                r#type: "number",
                                step: "0.1",
                                value: "{b21()}",
                                oninput: move |evt| {
                                    if let Ok(val) = evt.value().parse::<f64>() {
                                        b21.set(val);
                                    }
                                }
                            }
                            input {
                                r#type: "number",
                                step: "0.1",
                                value: "{b22()}",
                                oninput: move |evt| {
                                    if let Ok(val) = evt.value().parse::<f64>() {
                                        b22.set(val);
                                    }
                                }
                            }
                        }
                    }

                    div { class: "matrix-input-group",
                        h4 { "Vector v (2×1)" }
                        div { class: "vector-input-row",
                            input {
                                r#type: "number",
                                step: "0.1",
                                value: "{v1()}",
                                oninput: move |evt| {
                                    if let Ok(val) = evt.value().parse::<f64>() {
                                        v1.set(val);
                                    }
                                }
                            }
                            input {
                                r#type: "number",
                                step: "0.1",
                                value: "{v2()}",
                                oninput: move |evt| {
                                    if let Ok(val) = evt.value().parse::<f64>() {
                                        v2.set(val);
                                    }
                                }
                            }
                        }
                    }
                }

                // Right panel: Results
                div { class: "matrix-panel results-panel",
                    h3 { "Operations & Results" }

                    div { class: "operation-result",
                        h4 { "Matrix Addition: A + B" }
                        div { class: "result-matrix",
                            {
                                let result = result_add();
                                rsx! {
                                    div { class: "matrix-display",
                                        span { "[" }
                                        span { "{result[(0,0)]:.1}" }
                                        span { "{result[(0,1)]:.1}" }
                                        span { "]" }
                                    }
                                    div { class: "matrix-display",
                                        span { "[" }
                                        span { "{result[(1,0)]:.1}" }
                                        span { "{result[(1,1)]:.1}" }
                                        span { "]" }
                                    }
                                }
                            }
                        }
                    }

                    div { class: "operation-result",
                        h4 { "Matrix Multiplication: A × B" }
                        div { class: "result-matrix",
                            {
                                let result = result_mul();
                                rsx! {
                                    div { class: "matrix-display",
                                        span { "[" }
                                        span { "{result[(0,0)]:.1}" }
                                        span { "{result[(0,1)]:.1}" }
                                        span { "]" }
                                    }
                                    div { class: "matrix-display",
                                        span { "[" }
                                        span { "{result[(1,0)]:.1}" }
                                        span { "{result[(1,1)]:.1}" }
                                        span { "]" }
                                    }
                                }
                            }
                        }
                    }

                    div { class: "operation-result",
                        h4 { "Transpose: A" }
                        sup { "T" }
                        div { class: "result-matrix",
                            {
                                let result = result_transpose();
                                rsx! {
                                    div { class: "matrix-display",
                                        span { "[" }
                                        span { "{result[(0,0)]:.1}" }
                                        span { "{result[(0,1)]:.1}" }
                                        span { "]" }
                                    }
                                    div { class: "matrix-display",
                                        span { "[" }
                                        span { "{result[(1,0)]:.1}" }
                                        span { "{result[(1,1)]:.1}" }
                                        span { "]" }
                                    }
                                }
                            }
                        }
                    }

                    div { class: "operation-result",
                        h4 { "Matrix-Vector Product: A × v" }
                        div { class: "result-vector",
                            {
                                let result = result_mat_vec();
                                rsx! {
                                    div { class: "vector-display",
                                        span { "[" }
                                        span { "{result.data[0]:.1}" }
                                        span { "]" }
                                    }
                                    div { class: "vector-display",
                                        span { "[" }
                                        span { "{result.data[1]:.1}" }
                                        span { "]" }
                                    }
                                }
                            }
                        }
                    }

                    div { class: "code-hint",
                        code { "use linear_algebra::matrix::Matrix;" }
                        br {}
                        code { "let A = Matrix::from_vec(vec![...], 2, 2)?;" }
                        br {}
                        code { "let result = A + B;  // or A * B" }
                    }
                }
            }
        }
    }
}

#[component]
fn GradientDescentDemo() -> Element {
    // Training state
    let mut training_data = use_signal(|| vec![(1.0, 3.0), (2.0, 5.0), (3.0, 7.0), (4.0, 9.0), (5.0, 11.0)]);
    let mut is_training = use_signal(|| false);
    let mut trained_model = use_signal(|| None::<LinearRegressor>);
    let mut training_progress = use_signal(|| Vec::<f64>::new());
    let mut learning_rate = use_signal(|| 0.01);
    let mut iterations = use_signal(|| 500);

    // Input for new data point
    let mut new_x = use_signal(|| String::from("6"));
    let mut new_y = use_signal(|| String::from("13"));

    let train_model = move |_| {
        is_training.set(true);

        // Prepare training data
        let data = training_data();
        if data.is_empty() {
            is_training.set(false);
            return;
        }

        let x_vals: Vec<f64> = data.iter().map(|(x, _)| *x).collect();
        let y_vals: Vec<f64> = data.iter().map(|(_, y)| *y).collect();

        // Create matrix and vector
        let n = x_vals.len();
        let X = Matrix::from_vec(x_vals, n, 1).unwrap();
        let y = Vector { data: y_vals };

        // Train model
        let mut model = LinearRegressor::new(learning_rate());
        model.fit(&X, &y, iterations());

        // Store results
        training_progress.set(model.training_history.clone());
        trained_model.set(Some(model));
        is_training.set(false);
    };

    let add_data_point = move |_| {
        if let (Ok(x), Ok(y)) = (new_x().parse::<f64>(), new_y().parse::<f64>()) {
            training_data.write().push((x, y));
            trained_model.set(None); // Reset model when data changes
        }
    };

    let mut load_preset = move |preset: &str| {
        let data = match preset {
            "linear" => vec![(1.0, 3.0), (2.0, 5.0), (3.0, 7.0), (4.0, 9.0), (5.0, 11.0)],
            "steep" => vec![(1.0, 5.0), (2.0, 10.0), (3.0, 15.0), (4.0, 20.0)],
            "noisy" => vec![(1.0, 3.2), (2.0, 4.8), (3.0, 7.1), (4.0, 8.9), (5.0, 11.3)],
            _ => vec![],
        };
        training_data.set(data);
        trained_model.set(None);
    };

    let clear_data = move |_| {
        training_data.set(vec![]);
        trained_model.set(None);
        training_progress.set(vec![]);
    };

    rsx! {
        section { class: "demo-section gradient-descent",
            h2 { "🎯 Interactive Gradient Descent Trainer" }
            p { "Train a linear regression model and watch it learn in real-time!" }

            div { class: "trainer-container",
                // Left side: Data input
                div { class: "trainer-panel",
                    h3 { "Training Data" }

                    div { class: "preset-buttons",
                        button {
                            onclick: move |_| load_preset("linear"),
                            "📈 Linear (y=2x+1)"
                        }
                        button {
                            onclick: move |_| load_preset("steep"),
                            "📊 Steep (y=5x)"
                        }
                        button {
                            onclick: move |_| load_preset("noisy"),
                            "📉 Noisy"
                        }
                        button {
                            onclick: clear_data,
                            class: "danger",
                            "🗑️ Clear"
                        }
                    }

                    div { class: "data-list",
                        if training_data().is_empty() {
                            p { class: "empty-state", "No data points. Add some or load a preset!" }
                        } else {
                            table {
                                thead {
                                    tr {
                                        th { "x" }
                                        th { "y" }
                                        th { "Actions" }
                                    }
                                }
                                tbody {
                                    for (idx, (x, y)) in training_data().iter().enumerate() {
                                        tr {
                                            td { "{x}" }
                                            td { "{y}" }
                                            td {
                                                button {
                                                    onclick: move |_| {
                                                        training_data.write().remove(idx);
                                                        trained_model.set(None);
                                                    },
                                                    class: "small danger",
                                                    "✕"
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    div { class: "add-point",
                        h4 { "Add Data Point" }
                        div { class: "input-row",
                            input {
                                r#type: "number",
                                step: "0.1",
                                placeholder: "x",
                                value: "{new_x()}",
                                oninput: move |evt| new_x.set(evt.value())
                            }
                            input {
                                r#type: "number",
                                step: "0.1",
                                placeholder: "y",
                                value: "{new_y()}",
                                oninput: move |evt| new_y.set(evt.value())
                            }
                            button {
                                onclick: add_data_point,
                                "➕ Add"
                            }
                        }
                    }
                }

                // Middle: Training controls
                div { class: "trainer-panel",
                    h3 { "Training Parameters" }

                    div { class: "param-group",
                        label { "Learning Rate" }
                        input {
                            r#type: "number",
                            step: "0.001",
                            min: "0.001",
                            max: "1",
                            value: "{learning_rate()}",
                            oninput: move |evt| {
                                if let Ok(val) = evt.value().parse::<f64>() {
                                    learning_rate.set(val);
                                }
                            }
                        }
                        span { class: "param-value", "{learning_rate()}" }
                    }

                    div { class: "param-group",
                        label { "Iterations" }
                        input {
                            r#type: "number",
                            step: "100",
                            min: "100",
                            max: "5000",
                            value: "{iterations()}",
                            oninput: move |evt| {
                                if let Ok(val) = evt.value().parse::<usize>() {
                                    iterations.set(val);
                                }
                            }
                        }
                        span { class: "param-value", "{iterations()}" }
                    }

                    button {
                        onclick: train_model,
                        disabled: training_data().is_empty() || is_training(),
                        class: "train-button",
                        if is_training() {
                            "⏳ Training..."
                        } else {
                            "🚀 Train Model"
                        }
                    }

                    if let Some(model) = trained_model() {
                        div { class: "model-results",
                            h4 { "Learned Parameters" }
                            div { class: "param-display",
                                div { class: "param-row",
                                    span { class: "param-label", "Weight:" }
                                    span { class: "param-val learned", "{model.weights.data[0]:.6}" }
                                }
                                div { class: "param-row",
                                    span { class: "param-label", "Bias:" }
                                    span { class: "param-val learned", "{model.bias:.6}" }
                                }
                                div { class: "param-row",
                                    span { class: "param-label", "Final Cost:" }
                                    span { class: "param-val cost",
                                        "{model.training_history.last().unwrap_or(&0.0):.6}"
                                    }
                                }
                            }

                            // True parameters hint for y=2x+1
                            if training_data().len() >= 3 {
                                div { class: "hint",
                                    "💡 For y = 2x + 1: Weight ≈ 2.0, Bias ≈ 1.0"
                                }
                            }
                        }
                    }
                }

                // Right side: Cost visualization
                div { class: "trainer-panel",
                    h3 { "Training Progress" }

                    if !training_progress().is_empty() {
                        div { class: "cost-chart",
                            div { class: "chart-header",
                                span { "Cost over Iterations" }
                                span { class: "cost-reduction",
                                    {
                                        let history = training_progress();
                                        let start = history[0];
                                        let end = history[history.len()-1];
                                        let reduction = ((start - end) / start * 100.0).min(99.99);
                                        format!("↓ {:.1}% reduction", reduction)
                                    }
                                }
                            }

                            div { class: "cost-values",
                                {
                                    let history = training_progress();
                                    let samples: Vec<(usize, f64)> = history
                                        .iter()
                                        .enumerate()
                                        .step_by((history.len() / 10).max(1))
                                        .take(10)
                                        .map(|(i, &cost)| (i, cost))
                                        .collect();

                                    rsx! {
                                        for (iter, cost) in samples {
                                            div { class: "cost-item",
                                                span { class: "iter", "Iter {iter}:" }
                                                span { class: "cost", "{cost:.6}" }
                                            }
                                        }
                                    }
                                }
                            }

                            // Simple text-based "chart"
                            div { class: "ascii-chart",
                                {
                                    let history = training_progress();
                                    if !history.is_empty() {
                                        let max_cost = history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                        let min_cost = history.iter().cloned().fold(f64::INFINITY, f64::min);
                                        let range = max_cost - min_cost;

                                        rsx! {
                                            for (i, cost) in history.iter().enumerate().step_by((history.len() / 20).max(1)) {
                                                div { class: "chart-bar",
                                                    style: {
                                                        let height = if range > 0.0 {
                                                            ((cost - min_cost) / range * 100.0).max(2.0)
                                                        } else {
                                                            50.0
                                                        };
                                                        format!("height: {:.0}%", height)
                                                    },
                                                    title: "Iteration {i}: Cost = {cost:.6}"
                                                }
                                            }
                                        }
                                    } else {
                                        rsx! { div { "Train a model to see progress!" } }
                                    }
                                }
                            }
                        }
                    } else {
                        div { class: "empty-chart",
                            p { "📊 Cost chart will appear here after training" }
                            p { class: "hint", "Watch the gradient descent algorithm minimize the cost function!" }
                        }
                    }
                }
            }

            // Scatter Plot Visualization
            if let Some(model) = trained_model() {
                if !training_data().is_empty() {
                    div { class: "scatter-plot-container",
                        h3 { "📈 Data Visualization" }
                        p { class: "subtitle", "Data points and learned regression line" }

                        div { class: "plot-wrapper",
                            {
                                let data = training_data();

                                // Find data range for scaling
                                let x_vals: Vec<f64> = data.iter().map(|(x, _)| *x).collect();
                                let y_vals: Vec<f64> = data.iter().map(|(_, y)| *y).collect();

                                let x_min = x_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                                let x_max = x_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                let y_min = y_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                                let y_max = y_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                                // Add padding to ranges
                                let x_range = (x_max - x_min).max(1.0);
                                let y_range = (y_max - y_min).max(1.0);
                                let x_pad = x_range * 0.1;
                                let y_pad = y_range * 0.1;

                                let plot_x_min = x_min - x_pad;
                                let plot_x_max = x_max + x_pad;
                                let plot_y_min = y_min - y_pad;
                                let plot_y_max = y_max + y_pad;

                                let plot_width = plot_x_max - plot_x_min;
                                let plot_height = plot_y_max - plot_y_min;

                                // Helper function to convert data coords to SVG coords
                                let to_svg_x = |x: f64| -> f64 {
                                    (x - plot_x_min) / plot_width * 100.0
                                };
                                let to_svg_y = |y: f64| -> f64 {
                                    100.0 - (y - plot_y_min) / plot_height * 100.0
                                };

                                rsx! {
                                    svg {
                                        class: "scatter-plot",
                                        view_box: "0 0 100 100",
                                        xmlns: "http://www.w3.org/2000/svg",

                                        // Grid lines
                                        g { class: "grid",
                                            // Horizontal grid lines
                                            for i in 0..5 {
                                                line {
                                                    x1: "0",
                                                    y1: "{i * 25}",
                                                    x2: "100",
                                                    y2: "{i * 25}",
                                                    stroke: "#e5e7eb",
                                                    stroke_width: "0.2"
                                                }
                                            }
                                            // Vertical grid lines
                                            for i in 0..5 {
                                                line {
                                                    x1: "{i * 25}",
                                                    y1: "0",
                                                    x2: "{i * 25}",
                                                    y2: "100",
                                                    stroke: "#e5e7eb",
                                                    stroke_width: "0.2"
                                                }
                                            }
                                        }

                                        // Regression line
                                        g { class: "regression-line",
                                            {
                                                let w = model.weights.data[0];
                                                let b = model.bias;

                                                // Calculate line endpoints
                                                let line_x1 = plot_x_min;
                                                let line_y1 = w * line_x1 + b;
                                                let line_x2 = plot_x_max;
                                                let line_y2 = w * line_x2 + b;

                                                let svg_x1 = to_svg_x(line_x1);
                                                let svg_y1 = to_svg_y(line_y1);
                                                let svg_x2 = to_svg_x(line_x2);
                                                let svg_y2 = to_svg_y(line_y2);

                                                rsx! {
                                                    line {
                                                        x1: "{svg_x1}",
                                                        y1: "{svg_y1}",
                                                        x2: "{svg_x2}",
                                                        y2: "{svg_y2}",
                                                        stroke: "#2563eb",
                                                        stroke_width: "0.5",
                                                        stroke_dasharray: "2,1"
                                                    }
                                                }
                                            }
                                        }

                                        // Data points
                                        g { class: "data-points",
                                            for (x, y) in data.clone() {
                                                circle {
                                                    cx: "{to_svg_x(x)}",
                                                    cy: "{to_svg_y(y)}",
                                                    r: "2",
                                                    fill: "#10b981",
                                                    stroke: "#065f46",
                                                    stroke_width: "0.3"
                                                }
                                            }
                                        }

                                        // Predictions (for comparison)
                                        g { class: "predictions",
                                            {
                                                let w = model.weights.data[0];
                                                let b = model.bias;

                                                rsx! {
                                                    for (x, _y) in data {
                                                        {
                                                            let pred_y = w * x + b;
                                                            rsx! {
                                                                circle {
                                                                    cx: "{to_svg_x(x)}",
                                                                    cy: "{to_svg_y(pred_y)}",
                                                                    r: "1.5",
                                                                    fill: "#ef4444",
                                                                    opacity: "0.6"
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    div { class: "plot-legend",
                                        div { class: "legend-item",
                                            span { class: "legend-dot actual", "●" }
                                            span { "Actual data" }
                                        }
                                        div { class: "legend-item",
                                            span { class: "legend-dot predicted", "●" }
                                            span { "Predictions" }
                                        }
                                        div { class: "legend-item",
                                            span { class: "legend-line", "─" }
                                            span { "Regression line: y = {model.weights.data[0]:.2}x + {model.bias:.2}" }
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
