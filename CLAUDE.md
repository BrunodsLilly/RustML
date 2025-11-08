# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust workspace containing machine learning libraries, Python bindings, and a web application built with Dioxus. The project implements various machine learning algorithms from scratch, with a focus on linear algebra and regression models.

## Workspace Structure

The workspace follows a modular architecture where core libraries are shared across multiple binaries and applications:

**Core Libraries:**
- `linear_algebra/` - Foundational linear algebra operations and data structures (vectors, matrices)
- `loader/` - Data loading and parsing utilities
- `notes/` - Internal notes and documentation utilities

**ML Implementations:**
- `linear_regression/` - Linear regression library with examples under `examples/`
- `coursera_ml/` - Machine learning implementations from Coursera ML course
- `datasets/` - Dataset storage and management

**Visualization:**
- `plotting/` - Custom plotting library built on Plotters with binary examples in `src/bin/`

**Python Interop:**
- `pyml/` - Legacy Python bindings using PyO3 and Maturin (crate-type: cdylib)
- `python_bindings/` - Python bindings exposing core ML functionality as `coreml` module

**Web Application:**
- `web/` - Dioxus-based web application with routing support
  - Components in `src/components/` (nav, view)
  - Routes defined in `src/routes.rs`
  - Assets in `assets/` directory

## Build Commands

### Building the Workspace
```bash
cargo build                          # Build all workspace members
cargo build -p <package_name>        # Build specific package
cargo build --release                # Release build
```

### Running Examples
```bash
cargo run --example linear_regression_with_one_variable -p linear_regression
```

### Running Binaries
```bash
cargo run --bin example -p plotting    # Run plotting example
cargo run --bin mesh -p plotting       # Run mesh visualization
```

### Web Development
The web application uses Dioxus CLI (`dx`):

```bash
cd web
dx serve                               # Serve with default platform (web)
dx serve --platform desktop            # Desktop platform
dx serve --platform mobile             # Mobile platform
```

### Python Bindings

**pyml (legacy):**
```bash
cd pyml
maturin develop                        # Install in development mode
maturin build                          # Build wheel
```

**python_bindings (current):**
```bash
cd python_bindings
maturin develop                        # Install coreml module
```

### Testing
```bash
cargo test                             # Run all tests
cargo test -p <package_name>           # Test specific package
```

## Architecture Notes

### Dependency Flow
- `linear_regression` depends on `linear_algebra` and `loader`
- `coursera_ml` depends on `loader` and `notes`
- `python_bindings` exposes `loader` functionality to Python
- ML crates use `notes` as dev-dependency for documentation

### Build Profiles
The workspace defines custom profiles in root `Cargo.toml`:
- `wasm-dev` - Optimized WASM development (opt-level = 1)
- `server-dev` - Server development profile
- `android-dev` - Android development profile

### Python Integration
Two separate Python binding approaches exist:
1. **pyml** - Older implementation using PyO3 0.18.1 and Maturin 0.14
2. **python_bindings** - Current implementation using PyO3 0.25.1, exposes `coreml` module

### Web Application
- Built with Dioxus 0.6.0 router
- Supports multiple platforms via feature flags (web, desktop, mobile)
- Configuration in `Dioxus.toml` and linting rules in `clippy.toml`

## Edition Note

Most crates use `edition = "2024"` in their Cargo.toml files. This is likely a typo and should be "2021" (as Rust editions are released every 3 years: 2015, 2018, 2021, 2024 planned).
- emphasize client side performance because ee use WASM and can differentiate ourselves