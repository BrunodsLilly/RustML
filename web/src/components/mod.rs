// mod algorithm_arena; // TODO: Fix rsx! compilation issue
mod algorithm_arena_simple;
mod coefficient_display;
mod correlation_heatmap;
mod csv_upload;
mod feature_importance;
mod linear_regression_visualizer;
mod loss_functions;
mod ml_playground;
mod nav;
mod optimizer_demo;
mod optimizer_race_controller;
pub mod shared;
mod showcase;
mod view;

// pub use algorithm_arena::*; // TODO: Fix rsx! compilation issue
pub use algorithm_arena_simple::*;
pub use coefficient_display::*;
pub use correlation_heatmap::*;
pub use csv_upload::*;
pub use feature_importance::*;
pub use linear_regression_visualizer::*;
pub use loss_functions::*;
pub use ml_playground::*;
pub use nav::*;
pub use optimizer_demo::*;
pub use optimizer_race_controller::*;
pub use shared::*;
pub use showcase::*;
pub use view::*;
