//! ML Algorithm Arena - 3D Optimizer Racing Visualization
//!
//! This component creates an interactive 3D visualization where 4 optimization algorithms
//! (SGD, Momentum, RMSprop, Adam) compete in real-time to find the minimum of loss functions.
//!
//! **Vision:** Educational ML tool that feels like watching Formula 1 racing.
//! **Tech Stack:** Dioxus + three-d WebGL for 60 FPS 3D rendering in browser.
//!
//! **Performance Targets:**
//! - 500+ combined iterations/sec across 4 optimizers
//! - 60 FPS 3D rendering
//! - Interactive rotation/zoom with mouse
//! - Zero backend - all computation in WASM

use dioxus::prelude::*;

/// Main Arena component - orchestrates the entire 3D racing experience
#[component]
pub fn AlgorithmArena() -> Element {
    // State management
    let mut is_racing = use_signal(|| false);
    let mut selected_loss_fn = use_signal(|| String::from("Rosenbrock"));
    let mut learning_rate = use_signal(|| 0.01);
    let mut race_speed = use_signal(|| 500); // iterations per second

    rsx! {
        div { class: "arena-container",
            // Header
            header { class: "arena-header",
                h1 { "🏁 ML Algorithm Arena" }
                p { class: "tagline",
                    "Watch optimization algorithms race in real-time 3D"
                }
            }

            // Main 3D visualization area (placeholder for now)
            div { class: "arena-3d-container",
                id: "webgl-canvas-container",
                p { class: "placeholder-text",
                    "3D WebGL visualization will render here"
                }
                p { class: "tech-note",
                    "Powered by three-d + Rust WASM for 60 FPS real-time rendering"
                }
            }

            // Controls panel
            div { class: "arena-controls",
                div { class: "control-group",
                    label { "Loss Function:" }
                    select {
                        value: "{selected_loss_fn}",
                        onchange: move |evt| {
                            selected_loss_fn.set(evt.value());
                        },
                        option { value: "Rosenbrock", "Rosenbrock Valley" }
                        option { value: "Beale", "Beale Function" }
                        option { value: "Himmelblau", "Himmelblau Function" }
                    }
                }

                div { class: "control-group",
                    label { "Learning Rate: {learning_rate:.3}" }
                    input {
                        r#type: "range",
                        min: "0.001",
                        max: "0.1",
                        step: "0.001",
                        value: "{learning_rate}",
                        oninput: move |evt| {
                            if let Ok(val) = evt.value().parse::<f64>() {
                                learning_rate.set(val);
                            }
                        }
                    }
                }

                div { class: "control-group",
                    label { "Speed: {race_speed} iter/sec" }
                    input {
                        r#type: "range",
                        min: "100",
                        max: "1000",
                        step: "100",
                        value: "{race_speed}",
                        oninput: move |evt| {
                            if let Ok(val) = evt.value().parse::<usize>() {
                                race_speed.set(val);
                            }
                        }
                    }
                }

                button {
                    class: if *is_racing.read() { "btn-stop" } else { "btn-start" },
                    onclick: move |_| {
                        let current = *is_racing.read();
                        is_racing.set(!current);
                    },
                    if *is_racing.read() {
                        "⏸️ Pause Race"
                    } else {
                        "🏁 Start Race"
                    }
                }
            }

            // Leaderboard placeholder
            div { class: "arena-leaderboard",
                h3 { "🏆 Live Leaderboard" }
                div { class: "leaderboard-item first-place",
                    span { class: "rank", "1st 🏆" }
                    span { class: "optimizer-name", "Adam" }
                    span { class: "stats", "Loss: 0.003 | Iter: 142" }
                }
                div { class: "leaderboard-item",
                    span { class: "rank", "2nd 🥈" }
                    span { class: "optimizer-name", "RMSprop" }
                    span { class: "stats", "Loss: 0.021 | Iter: 156" }
                }
                div { class: "leaderboard-item",
                    span { class: "rank", "3rd 🥉" }
                    span { class: "optimizer-name", "Momentum" }
                    span { class: "stats", "Loss: 0.089 | Iter: 178" }
                }
                div { class: "leaderboard-item",
                    span { class: "rank", "4th" }
                    span { class: "optimizer-name", "SGD" }
                    span { class: "stats", "Loss: 0.234 | Iter: 201" }
                }
            }

            // Educational tooltip
            div { class: "arena-tooltip",
                h4 { "💡 Why is Adam winning?" }
                p {
                    "Adam adapts its learning rate per-parameter using momentum and RMSprop. "
                    "Watch how it navigates the narrow valley more efficiently than SGD!"
                }
            }
        }

        // Inline styles (will move to main.css later)
        style {
            r#"
            .arena-container {{
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
                padding: 2rem;
                max-width: 1400px;
                margin: 0 auto;
            }}

            .arena-header {{
                text-align: center;
            }}

            .arena-header h1 {{
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}

            .tagline {{
                font-size: 1.1rem;
                color: #666;
            }}

            .arena-3d-container {{
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                border-radius: 12px;
                min-height: 500px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }}

            .placeholder-text {{
                color: white;
                font-size: 1.5rem;
                margin-bottom: 0.5rem;
            }}

            .tech-note {{
                color: rgba(255,255,255,0.7);
                font-size: 0.9rem;
            }}

            .arena-controls {{
                display: flex;
                gap: 1.5rem;
                align-items: center;
                flex-wrap: wrap;
                padding: 1.5rem;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}

            .control-group {{
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }}

            .control-group label {{
                font-weight: 600;
                color: #333;
                font-size: 0.9rem;
            }}

            .control-group select,
            .control-group input[type="range"] {{
                padding: 0.5rem;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                font-size: 1rem;
            }}

            .btn-start, .btn-stop {{
                padding: 0.75rem 2rem;
                font-size: 1.1rem;
                font-weight: 600;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s ease;
            }}

            .btn-start {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}

            .btn-start:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            }}

            .btn-stop {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
            }}

            .arena-leaderboard {{
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}

            .arena-leaderboard h3 {{
                margin-bottom: 1rem;
                color: #333;
            }}

            .leaderboard-item {{
                display: flex;
                align-items: center;
                gap: 1rem;
                padding: 0.75rem;
                border-radius: 8px;
                margin-bottom: 0.5rem;
                background: #f8f9fa;
            }}

            .leaderboard-item.first-place {{
                background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
                font-weight: 600;
            }}

            .rank {{
                font-size: 1.2rem;
                min-width: 60px;
            }}

            .optimizer-name {{
                flex: 1;
                font-weight: 600;
                color: #333;
            }}

            .stats {{
                color: #666;
                font-size: 0.9rem;
            }}

            .arena-tooltip {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}

            .arena-tooltip h4 {{
                margin-bottom: 0.5rem;
            }}

            .arena-tooltip p {{
                line-height: 1.6;
            }}
            "#
        }
    }
}
