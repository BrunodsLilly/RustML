# Bruno's Machine Learning Workspace

[![CI](https://github.com/brunodossantos/RustML-cicd/actions/workflows/ci.yml/badge.svg)](https://github.com/brunodossantos/RustML-cicd/actions/workflows/ci.yml)
[![Performance](https://github.com/brunodossantos/RustML-cicd/actions/workflows/performance.yml/badge.svg)](https://github.com/brunodossantos/RustML-cicd/actions/workflows/performance.yml)

A Rust-based machine learning workspace with Python bindings and a web-based visualization interface built with Dioxus.

## Overview

This workspace contains:
- **Core ML Libraries**: Linear algebra, linear regression, neural networks, and data loading utilities written in Rust
- **Neural Network**: Complete backpropagation implementation with multiple activation functions and weight initialization strategies
- **Python Bindings**: PyO3-based bindings exposing ML functionality to Python via the `coreml` module
- **Web Application**: Dioxus-powered web app for interactive ML visualization and experimentation
- **Plotting**: Custom plotting library built on Plotters for data visualization

## 🚀 What's New

**Neural Network with Backpropagation!** 🧠 Full-featured multi-layer perceptron implementation:

- **Complete Backpropagation**: Automatic gradient computation through all layers
- **Multiple Activation Functions**: Sigmoid, ReLU, Tanh, and Linear activations
- **Smart Weight Initialization**: Xavier (Glorot) and He initialization strategies
- **Training History**: Track losses, accuracies, and network snapshots
- **XOR Problem Solved**: Successfully learns non-linearly separable functions
- **23 Comprehensive Tests**: All passing with 100% accuracy on XOR demo

```rust
use neural_network::{NeuralNetwork, activation::ActivationType};
use linear_algebra::matrix::Matrix;

// XOR training data
let X = Matrix::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], 4, 2).unwrap();
let y = Matrix::from_vec(vec![0.0, 1.0, 1.0, 0.0], 4, 1).unwrap();

// Create 2-4-1 network (2 inputs, 4 hidden, 1 output)
let mut nn = NeuralNetwork::new(
    &[2, 4, 1],
    &[ActivationType::Tanh, ActivationType::Sigmoid],
    0.5,
);

// Train with backpropagation
nn.fit(&X, &y, 1000, 0);

// Predictions: [0.013, 0.971, 0.971, 0.035] - Perfect XOR!
```

**Interactive ML Playground is Live!** 🎉 Train models in your browser with real-time visualizations:

- **Matrix Operations Calculator**: Interactive 2×2 matrix arithmetic with instant results
- **Gradient Descent Trainer**: Watch linear regression learn from data in real-time
- **Scatter Plot Visualization**: See data points, predictions, and regression lines dynamically
- **Cost Function Graphs**: Animated bar charts showing convergence during training

**Fully functional gradient descent linear regression!** Train models from scratch in pure Rust:

```rust
use linear_algebra::{matrix::Matrix, vectors::Vector};
use linear_regression::LinearRegressor;

// Create training data: y = 2x + 1
let X = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], 5, 1).unwrap();
let y = Vector { data: vec![3.0, 5.0, 7.0, 9.0, 11.0] };

// Train with gradient descent
let mut model = LinearRegressor::new(0.01);
model.fit(&X, &y, 1000);

// Learned weight: 2.01, bias: 0.91 (true: 2.0, 1.0)
// Final cost: 0.0008
```

**Complete Matrix operations with an intuitive API:**

```rust
use linear_algebra::matrix;

let m = matrix![
    [1, 2, 3],
    [4, 5, 6]
];

let transposed = m.transpose();
let col = m.col(1).unwrap();  // [2, 5]
```

**Interactive web showcase with real-time ML demos** at `/showcase`!

## Quick Start

### Prerequisites

- Rust (latest stable)
- [Dioxus CLI](https://dioxuslabs.com/learn/0.6/CLI) (`cargo install dioxus-cli`)
- Python 3.7+ with pip (for Python bindings)
- [Maturin](https://www.maturin.rs/) (`pip install maturin`) (for Python bindings)

### Running the Web App

```bash
cd web
dx serve                               # Development server at http://localhost:8080
dx serve --platform desktop            # Run as desktop app
```

The web app currently provides:
- **Main View** (`/`) - Landing page with navigation to various ML modules
- **ML Library Showcase** (`/showcase`) - Interactive demonstrations of the ML libraries
  - **Vector Operations**: Addition, scalar multiplication, dot product with live examples
  - **Interactive Vector Calculator**: Real-time vector arithmetic with adjustable inputs
  - **Interactive Matrix Operations**: Real-time 2×2 matrix operations including:
    - Matrix addition (A + B)
    - Matrix multiplication (A × B)
    - Matrix transpose (A^T)
    - Matrix-vector multiplication (A × v)
  - **Gradient Descent Trainer**: Complete interactive ML training experience
    - Load preset datasets or create custom data points
    - Adjust hyperparameters (learning rate, iterations)
    - Train linear regression models in real-time
    - Visualize cost reduction over iterations with bar charts
    - **NEW**: Scatter plot visualization showing data points, predictions, and learned regression line
- **Courses View** (`/courses`) - ML course materials and implementations

### Building & Testing

```bash
# Build entire workspace
cargo build

# Run tests
cargo test

# Build specific package
cargo build -p linear_regression

# Run linear regression example (trains on y = 2x + 1)
cargo run --example linear_regression_with_one_variable -p linear_regression
# Output shows gradient descent converging from cost 88.5 → 0.0008!

# Run neural network XOR demo (learns XOR function)
cargo run --example xor_demo -p neural_network
# Output shows network achieving 100% accuracy on XOR problem!
```

### CI/CD Pipeline

This project uses automated testing on all pull requests:

- **Rust Tests**: All unit tests, formatting (rustfmt), and linting (clippy)
- **WASM Build**: Web application compilation and bundle size checks
- **Playwright E2E**: Browser-based integration tests
- **Security Audit**: Dependency vulnerability scanning
- **Performance Benchmarks**: Optimizer performance validation

**Pre-push validation:**
```bash
.github/scripts/pre-push.sh    # Run all checks locally before pushing
```

See [CI/CD Setup Documentation](docs/CI_CD_SETUP.md) for details.

### Python Integration

```bash
cd python_bindings
maturin develop              # Install coreml module in current Python environment
python
>>> import coreml
>>> # Use ML functions from Rust
```

## Architecture

### How the Web App Leverages Library Code

The web application is designed to provide an interactive interface for the ML libraries:

**Current Architecture:**
- `web/` - Dioxus frontend with routing and component system
  - Uses Dioxus Router for navigation between ML modules
  - Component-based architecture (`components/nav.rs`, `components/view.rs`, `components/showcase.rs`)
  - Static assets in `assets/` (CSS, icons)
  - **Now integrated with `linear_algebra` library for interactive demos**

**Integration Points:**
- ✅ **Implemented**: `linear_algebra` integration with interactive vector and matrix operations
- ✅ **Implemented**: `linear_regression` integration with real-time gradient descent training
- ✅ **Implemented**: Interactive parameter tuning for ML models (learning rate, iterations)
- ✅ **Implemented**: Real-time data visualization with scatter plots and cost charts
- ✅ **Implemented**: Custom data point creation and preset datasets
- 🚧 **Planned**: Data loading via `loader` crate for CSV/dataset handling
- 🚧 **Planned**: Integration with `plotting` library for advanced visualizations

**Library Dependencies:**
```
linear_regression
    ├─ linear_algebra (vectors, matrices)
    └─ loader (data I/O)

neural_network ✅
    └─ linear_algebra (vectors, matrices)

web (current)
    ├─ linear_algebra ✅
    ├─ linear_regression ✅
    ├─ neural_network 🚧 (planned)
    └─ loader ✅
```

### Current Library Status

**Implemented:**
- `linear_algebra` - Vector and matrix operations
- `loader` - File reading utilities
- `plotting` - Plotters-based visualization with binary examples

**In Development:**
- Web app integration with additional ML libraries
- Advanced interactive visualizations
- Plotting library integration

**Fully Implemented:**
- `linear_regression` - Complete gradient descent implementation
  - Batch prediction with Matrix operations
  - Mean Squared Error cost function
  - Gradient descent training (`fit()` and `fit_until_converged()`)
  - Training history tracking
  - Single and batch predictions
  - 8 comprehensive unit tests (all passing)
  - Working example demonstrating learning y = 2x + 1

- `neural_network` - Multi-layer perceptron with backpropagation
  - Forward and backward propagation
  - Multiple activation functions (Sigmoid, ReLU, Tanh, Linear)
  - Weight initialization strategies (Xavier, He, Zeros, SmallRandom)
  - Batch gradient descent training
  - Training history with snapshots for visualization
  - Mean Squared Error loss function
  - 23 comprehensive unit tests (all passing)
  - XOR problem demo achieving 100% accuracy

## Developer Experience Recommendations

### Recommended Setup

1. **Hot Reload Development:**
   ```bash
   cd web
   dx serve --hot-reload    # Auto-reload on code changes
   ```

2. **Watch Mode for Libraries:**
   ```bash
   cargo watch -x 'test -p linear_regression'
   cargo watch -x 'check --all'
   ```

3. **Parallel Development:**
   - Terminal 1: `dx serve` (web app)
   - Terminal 2: `cargo watch -x test` (library tests)
   - Terminal 3: Your editor

4. **Pre-commit Hooks:**
   ```bash
   # Add to .git/hooks/pre-commit
   #!/bin/bash
   cargo fmt --all -- --check
   cargo clippy --all -- -D warnings
   cargo test --all
   ```

### Suggested Tooling

- **IDE**: VSCode with rust-analyzer
- **Extensions**:
  - rust-analyzer
  - Even Better TOML
  - CodeLLDB (debugging)
  - Dioxus extension (if available)

- **Quality Tools**:
  ```bash
  cargo install cargo-watch     # Auto-rebuild on changes
  cargo install cargo-expand    # Expand macros
  cargo install cargo-edit      # Manage dependencies
  ```

### Development Workflow

1. **Feature Development**:
   ```bash
   # Create feature in library
   cd linear_regression
   cargo test

   # Integrate into web app
   cd ../web
   dx serve

   # Test Python bindings
   cd ../python_bindings
   maturin develop
   python test_script.py
   ```

2. **Performance Profiling**:
   ```bash
   cargo build --release -p linear_regression
   cargo bench  # (once benchmarks are added)
   ```

## TODO

### ✅ Recently Completed

- [x] **Matrix Operations Implementation**
  - ✅ Full Matrix struct with row-major storage
  - ✅ Matrix-matrix, matrix-vector, and scalar operations
  - ✅ Transpose, indexing, and slicing
  - ✅ Helper functions (zeros, ones, identity)
  - ✅ Convenient `matrix![]` macro
  - ✅ 12 comprehensive unit tests (all passing)

- [x] **LinearRegressor Implementation**
  - ✅ Batch gradient descent training
  - ✅ Early stopping with convergence threshold
  - ✅ Mean Squared Error cost function
  - ✅ Batch and single sample prediction
  - ✅ Training history tracking
  - ✅ 8 comprehensive unit tests including multivariate regression
  - ✅ Working example learning linear relationships

- [x] **Web Application Showcase**
  - ✅ Interactive vector calculator with real-time computation
  - ✅ Static vector operation demonstrations
  - ✅ Comprehensive Playwright E2E test suite (10 tests passing)
  - ✅ Clean UI with styled components

- [x] **Fix Cargo.toml Edition Field**
  - ✅ Changed `edition = "2024"` to `edition = "2021"` in all 8 packages
  - ✅ All packages now compile successfully

- [x] **Interactive Matrix Operations Demo** 🎉 NEW!
  - ✅ Real-time 2×2 matrix input with live computation
  - ✅ Matrix addition, multiplication, and transpose
  - ✅ Matrix-vector multiplication
  - ✅ Professional UI with orange/gold theming
  - ✅ All operations update instantly as inputs change

- [x] **Gradient Descent Trainer Integration** 🎉 NEW!
  - ✅ Interactive gradient descent visualization with 3-panel layout
  - ✅ Preset datasets (linear, steep, noisy) and custom data points
  - ✅ Adjustable hyperparameters (learning rate, iterations)
  - ✅ Real-time model training in the browser (WASM)
  - ✅ Cost reduction visualization with animated bar charts
  - ✅ Learned parameters display with hints
  - ✅ **Scatter plot visualization** showing data points, predictions, and regression line
  - ✅ Automatic scaling and responsive design

- [x] **Neural Network Implementation** 🧠 NEW!
  - ✅ Complete backpropagation algorithm with automatic gradient computation
  - ✅ Multi-layer perceptron supporting arbitrary network architectures
  - ✅ Four activation functions: Sigmoid, ReLU, Tanh, Linear
  - ✅ Smart weight initialization: Xavier (Glorot) and He strategies
  - ✅ Training history tracking (losses, accuracies, snapshots)
  - ✅ 23 comprehensive unit tests (all passing)
  - ✅ XOR problem demo achieving 100% accuracy
  - ✅ Modular design ready for web integration

### High Priority

- [ ] **Interactive Neural Network Visualizer**
  - Real-time network architecture diagram with animated neuron activations
  - Interactive layer-by-layer forward propagation visualization
  - Backpropagation animation showing gradient flow
  - XOR and circle classification demos
  - Adjustable network architecture (add/remove layers, neurons)
  - Live weight matrix heatmaps
  - Training progress charts and decision boundary plots

- [ ] **Advanced Visualizations**
  - Add polynomial regression demo
  - Multiple regression with 3D visualization
  - Regularization (L1/L2) interactive demo
  - Feature scaling visualization

### Medium Priority

- [ ] **Enhanced Data Visualization**
  - Integrate plotting library into web app
  - Add interactive charts (scatter plots, line plots, heatmaps)
  - Real-time visualization of training progress
  - Support for multiple datasets

- [ ] **Web App Features**
  - Add dataset management page
  - Model training interface with hyperparameter controls
  - Results comparison dashboard
  - Export trained models

- [ ] **Testing & Documentation**
  - Add comprehensive unit tests for LinearRegressor
  - Integration tests for web app
  - Add rustdoc comments to public APIs
  - Create API documentation site

- [ ] **Developer Experience**
  - Add cargo-make or just recipes for common tasks
  - ✅ Set up CI/CD pipeline (GitHub Actions)
  - ✅ Add pre-commit hooks (via pre-push script)
  - Create development containers/Docker setup

### Low Priority

- [ ] **Advanced ML Features**
  - Implement regularization (L1, L2)
  - Add polynomial regression
  - Support for multiple regression algorithms
  - Model serialization/deserialization

- [ ] **Python Bindings Enhancement**
  - Consolidate pyml and python_bindings into single implementation
  - Add NumPy integration
  - Create pip package
  - Add Python type stubs

- [ ] **Performance Optimization**
  - Add SIMD optimizations to linear algebra operations
  - Implement parallel training
  - Add benchmarking suite
  - Profile and optimize hot paths

- [ ] **Cross-Platform Support**
  - Test and optimize Android build profile
  - iOS support for mobile app
  - Desktop app packaging (native installers)

## License

[Add license information]

## Contributing

[Add contribution guidelines]
