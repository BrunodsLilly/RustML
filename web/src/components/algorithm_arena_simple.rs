//! ML Algorithm Arena - Simplified Version for Testing
//!
//! Minimal working version to debug compilation issues

use super::{loss_functions::LossFunction, optimizer_race_controller::RaceController};
use dioxus::prelude::*;

#[component]
pub fn AlgorithmArenaSimple() -> Element {
    let mut race_controller = use_signal(|| {
        RaceController::new(LossFunction::Rosenbrock, 0.01, (-0.5, 1.5))
    });

    rsx! {
        div { class: "arena-container",
            h1 { "🏁 ML Algorithm Arena" }

            button {
                onclick: move |_| {
                    race_controller.write().step_n(100);
                },
                "Step 100 Iterations"
            }

            button {
                onclick: move |_| {
                    race_controller.write().reset((-0.5, 1.5));
                },
                "Reset"
            }

            div { class: "leaderboard",
                h3 { "Leaderboard" }
                p { "Total Iterations: {race_controller.read().total_iterations}" }

                // Simple iteration over racers
                {race_controller.read().get_leaderboard().iter().map(|racer| {
                    rsx! {
                        div { key: "{racer.name}",
                            "{racer.rank}. {racer.name} - Loss: {racer.current_loss():.6}"
                        }
                    }
                })}
            }
        }
    }
}
