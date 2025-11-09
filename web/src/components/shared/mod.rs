/// Shared UI components for the ML Playground
///
/// This module contains reusable components that provide common functionality
/// across the application, including error handling, validation, and UI primitives.

pub mod error_boundary;

pub use error_boundary::{ErrorBoundary, catch_panic, catch_panic_async, validate_algorithm_input};
