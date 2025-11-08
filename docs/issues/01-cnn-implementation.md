# feat: Implement Convolutional Neural Networks (CNN) for Image Classification

## Overview

Add a comprehensive CNN implementation to enable real-time image classification in the browser. This will unlock MNIST digit recognition, demonstrate WASM's capability for computationally intensive ML, and provide an interactive educational tool for understanding how CNNs work.

**Priority:** 🔥 HIGHEST - This is the natural next step after optimizer visualization and unlocks the "killer app" for browser ML (image classification).

## Problem Statement / Motivation

### Current Limitations
- Only fully-connected neural networks supported
- No image-specific layers (convolution, pooling)
- Cannot efficiently process visual data
- Missing the most popular deep learning architecture

### Why This Matters
1. **Educational Impact:** CNNs are fundamental to modern ML - visualizing filters and activations provides instant understanding
2. **Performance Showcase:** Demonstrates WASM can handle real-time image processing at 60 FPS
3. **Practical Applications:** Digit recognition, emoji detection, simple object classification
4. **Market Differentiation:** No other browser-based ML tool offers interactive CNN training with visualization

### User Stories
- **As a student:** I want to draw a digit and see the CNN classify it in real-time to understand how CNNs work
- **As a developer:** I want to train a CNN on MNIST in my browser without backend infrastructure
- **As an educator:** I want to show students what each CNN layer "sees" through filter visualizations

## Proposed Solution

### High-Level Architecture

```
New Crate: cnn/
├─ src/
│  ├─ lib.rs              # Main CNN struct, training loop
│  ├─ layers/
│  │  ├─ mod.rs           # Layer trait definition
│  │  ├─ conv2d.rs        # Convolution layer (im2col approach)
│  │  ├─ pooling.rs       # MaxPool2D, AvgPool2D
│  │  ├─ flatten.rs       # Flatten for transition to dense
│  │  └─ dense.rs         # Wrapper for neural_network::Layer
│  ├─ activation.rs       # Reuse neural_network::activation
│  └─ initializer.rs      # He initialization for ReLU layers
├─ examples/
│  ├─ mnist_demo.rs       # CLI training on MNIST
│  └─ simple_cnn.rs       # Minimal example (3x3 image)
├─ tests/
│  └─ cnn_tests.rs        # Gradient checks, shape tests
└─ benches/
   └─ conv_bench.rs       # Benchmark convolution implementations
```

### CNN Architecture (LeNet-5 for MNIST)

```rust
// Target architecture - proven on MNIST (95%+ accuracy)
let cnn = CNN::builder()
    .input_shape((28, 28, 1))  // MNIST grayscale images
    .add_conv2d(6, (5, 5))      // 6 filters, 5x5 kernel → (24, 24, 6)
    .add_activation(ReLU)
    .add_maxpool2d((2, 2))      // → (12, 12, 6)
    .add_conv2d(16, (5, 5))     // → (8, 8, 16)
    .add_activation(ReLU)
    .add_maxpool2d((2, 2))      // → (4, 4, 16)
    .add_flatten()              // → 256 features
    .add_dense(120)             // Fully connected
    .add_activation(ReLU)
    .add_dense(84)
    .add_activation(ReLU)
    .add_dense(10)              // 10 digit classes
    .add_activation(Softmax)
    .build()?;
```

## Technical Approach

### Implementation Strategy

**Phase 1: Core CNN Library (Weeks 1-2)**

1. **Convolution Implementation:**
   - Use **im2col** approach (matrix multiplication via `linear_algebra`)
   - Zero-allocation variant for single-image inference: `forward_single()`
   - Batch processing for training efficiency

```rust
// cnn/src/layers/conv2d.rs
pub struct Conv2D {
    filters: usize,           // Number of output channels
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    weights: Matrix<f64>,     // Shape: (filters, kernel_h * kernel_w * in_channels)
    bias: Vector<f64>,        // Shape: (filters,)
    // Cached for backprop
    last_input: Option<Matrix<f64>>,
    im2col_cache: Matrix<f64>,
}

impl Conv2D {
    /// Forward pass using im2col (leverages fast matrix multiply)
    pub fn forward(&mut self, input: &Matrix<f64>) -> Result<Matrix<f64>, String> {
        // input shape: (batch, height, width, channels)
        // 1. im2col: rearrange input patches into columns
        let col = self.im2col(input)?;

        // 2. Matrix multiply: weights × col
        let output = &self.weights * &col;

        // 3. Add bias, reshape to (batch, out_h, out_w, filters)
        Ok(self.col2im(output))
    }

    /// Zero-allocation inference for single image (browser visualization)
    pub fn forward_single(&self, input: &[f64]) -> Vec<f64> {
        // Specialized path: no Matrix allocation, direct convolution
        // Similar pattern to Optimizer::step_2d()
    }
}
```

2. **Pooling Layers:**

```rust
// cnn/src/layers/pooling.rs
pub enum PoolingType {
    Max,
    Average,
}

pub struct Pooling2D {
    pool_size: (usize, usize),
    stride: (usize, usize),
    pool_type: PoolingType,
    // Cache for backprop
    max_indices: Option<Vec<(usize, usize)>>,
}

impl Pooling2D {
    pub fn forward(&mut self, input: &Matrix<f64>) -> Result<Matrix<f64>, String> {
        match self.pool_type {
            PoolingType::Max => self.max_pool(input),
            PoolingType::Average => self.avg_pool(input),
        }
    }

    pub fn backward(&self, grad_output: &Matrix<f64>) -> Result<Matrix<f64>, String> {
        // Use cached max_indices for gradient routing
    }
}
```

3. **Layer Trait for Polymorphism:**

```rust
// cnn/src/layers/mod.rs
pub trait Layer: std::fmt::Debug {
    fn forward(&mut self, input: &Matrix<f64>) -> Result<Matrix<f64>, String>;
    fn backward(&mut self, grad_output: &Matrix<f64>) -> Result<Matrix<f64>, String>;
    fn update_weights(&mut self, learning_rate: f64);
    fn output_shape(&self, input_shape: (usize, usize, usize)) -> (usize, usize, usize);
}
```

4. **CNN Builder Pattern:**

```rust
// cnn/src/lib.rs
pub struct CNN {
    layers: Vec<Box<dyn Layer>>,
    input_shape: (usize, usize, usize),  // (height, width, channels)
    learning_rate: f64,
}

impl CNN {
    pub fn builder() -> CNNBuilder {
        CNNBuilder::new()
    }

    pub fn forward(&mut self, input: &Matrix<f64>) -> Result<Matrix<f64>, String> {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output)?;
        }
        Ok(output)
    }

    pub fn backward(&mut self, loss_gradient: &Matrix<f64>) -> Result<(), String> {
        let mut grad = loss_gradient.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad)?;
        }
        Ok(())
    }

    pub fn update_weights(&mut self) {
        for layer in &mut self.layers {
            layer.update_weights(self.learning_rate);
        }
    }
}
```

**Phase 2: MNIST Integration (Week 3)**

5. **Dataset Loading:**

```rust
// cnn/src/mnist.rs
pub struct MNISTDataset {
    images: Matrix<f64>,      // (60000, 28, 28, 1)
    labels: Vec<usize>,       // (60000,) - digit classes
}

impl MNISTDataset {
    /// Load MNIST from embedded bytes (compiled into WASM)
    pub fn from_bytes(images: &[u8], labels: &[u8]) -> Result<Self, String> {
        // Parse IDX format
        // Normalize to [0, 1]
    }

    /// Get batch for training
    pub fn batch(&self, indices: &[usize]) -> (Matrix<f64>, Vec<usize>) {
        // Return (images, labels) for specified indices
    }
}
```

6. **Training Loop:**

```rust
// cnn/examples/mnist_demo.rs
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load data
    let train_data = MNISTDataset::from_bytes(
        include_bytes!("../data/train-images.idx"),
        include_bytes!("../data/train-labels.idx"),
    )?;

    let test_data = MNISTDataset::from_bytes(
        include_bytes!("../data/test-images.idx"),
        include_bytes!("../data/test-labels.idx"),
    )?;

    // Create CNN
    let mut cnn = CNN::builder()
        .input_shape((28, 28, 1))
        .add_conv2d(6, (5, 5))
        .add_activation(ReLU)
        .add_maxpool2d((2, 2))
        .add_conv2d(16, (5, 5))
        .add_activation(ReLU)
        .add_maxpool2d((2, 2))
        .add_flatten()
        .add_dense(120)
        .add_activation(ReLU)
        .add_dense(84)
        .add_activation(ReLU)
        .add_dense(10)
        .build()?;

    // Train
    let epochs = 10;
    let batch_size = 32;

    for epoch in 0..epochs {
        for batch in train_data.batches(batch_size) {
            let (images, labels) = batch;

            // Forward
            let predictions = cnn.forward(&images)?;

            // Loss (cross-entropy)
            let loss = cross_entropy_loss(&predictions, &labels);

            // Backward
            let loss_grad = cross_entropy_gradient(&predictions, &labels);
            cnn.backward(&loss_grad)?;

            // Update
            cnn.update_weights();
        }

        // Evaluate on test set
        let accuracy = evaluate(&cnn, &test_data);
        println!("Epoch {}: Accuracy = {:.2}%", epoch, accuracy * 100.0);
    }

    Ok(())
}
```

**Phase 3: Interactive Web Demo (Week 4)**

7. **Browser Integration:**

```rust
// web/src/components/cnn_demo.rs
#[component]
pub fn CNNDemo() -> Element {
    let mut model = use_signal(|| load_pretrained_cnn());
    let mut canvas_data = use_signal(|| vec![0u8; 28 * 28]);
    let mut prediction = use_signal(|| None::<(usize, f64)>);

    let predict = move |_| {
        // Get canvas pixels
        let pixels = canvas_data.read();

        // Preprocess (normalize, reshape)
        let input = preprocess(&pixels);

        // Inference (FAST PATH - zero allocations)
        let output = model.read().forward_single(&input);

        // Get top prediction
        let (digit, confidence) = argmax_with_confidence(&output);
        prediction.set(Some((digit, confidence)));
    };

    rsx! {
        div { class: "cnn-demo",
            // Drawing canvas (28x28)
            Canvas {
                width: 280,
                height: 280,
                on_draw: move |pixels| canvas_data.set(pixels),
            }

            button { onclick: predict, "Classify" }
            button { onclick: move |_| canvas_data.set(vec![0u8; 28*28]), "Clear" }

            // Prediction display
            if let Some((digit, conf)) = *prediction.read() {
                div { class: "prediction",
                    h2 { "Prediction: {digit}" }
                    p { "Confidence: {conf:.1}%" }
                }
            }

            // Filter visualization
            FilterVisualization { model: model.read() }

            // Activation maps
            ActivationVisualization { model: model.read(), input: canvas_data.read() }
        }
    }
}
```

8. **Visualization Components:**

```rust
// web/src/components/cnn_visualizations.rs

/// Visualize learned filters (what each neuron detects)
#[component]
fn FilterVisualization(model: CNN) -> Element {
    let first_layer_filters = model.layers[0].weights();  // (6, 5, 5, 1)

    rsx! {
        div { class: "filters",
            h3 { "Learned Filters (Layer 1)" }
            div { class: "filter-grid",
                for (i, filter) in first_layer_filters.iter().enumerate() {
                    Canvas {
                        key: "{i}",
                        width: 50,
                        height: 50,
                        pixels: normalize_for_display(filter),
                    }
                }
            }
        }
    }
}

/// Visualize activation maps (what the network "sees")
#[component]
fn ActivationVisualization(model: CNN, input: Vec<u8>) -> Element {
    let activations = model.get_layer_activations(&input);

    rsx! {
        div { class: "activations",
            h3 { "Activation Maps" }
            for (layer_name, activation_map) in activations {
                div { class: "layer-activations",
                    h4 { "{layer_name}" }
                    // Display each channel's activation
                    for channel in activation_map.channels() {
                        Canvas {
                            pixels: channel.to_heatmap(),
                        }
                    }
                }
            }
        }
    }
}
```

## Alternative Approaches Considered

### 1. Using Burn Framework
**Pros:**
- Production-ready CNN implementation
- WGPU backend for GPU acceleration
- Comprehensive training utilities

**Cons:**
- Large bundle size (2-3 MB WASM overhead)
- Less educational (black box)
- Harder to customize for visualization

**Decision:** Build from scratch for educational value, consider Burn for Phase 2 (advanced models).

### 2. Direct Convolution (Naive Approach)
**Pros:**
- Simple to understand
- No im2col overhead

**Cons:**
- 5-10x slower than im2col
- Cannot leverage optimized BLAS

**Decision:** Use im2col for training, add direct convolution only if needed for inference.

### 3. FFT-based Convolution
**Pros:**
- Faster for large kernels (11x11+)
- Theoretically optimal O(n log n)

**Cons:**
- Overkill for 3x3 and 5x5 kernels
- Complex to implement correctly
- Not worth the complexity for MNIST

**Decision:** Defer to future if implementing ImageNet-scale models.

## Acceptance Criteria

### Functional Requirements

#### Core CNN Library
- [ ] Conv2D layer supports arbitrary kernel sizes, strides, padding
- [ ] MaxPool2D and AvgPool2D layers implemented
- [ ] Flatten layer transitions from conv to dense
- [ ] Forward pass computes correct output shapes
- [ ] Backward pass computes correct gradients (verified via gradient checking)
- [ ] Builder pattern allows flexible architecture construction
- [ ] Supports multi-channel inputs (grayscale and RGB)

#### MNIST Training
- [ ] Can load MNIST dataset from embedded bytes in WASM
- [ ] Training loop converges to 95%+ test accuracy within 10 epochs
- [ ] Batch processing supported (batch_size configurable)
- [ ] Training progress displayed (loss per epoch, accuracy)
- [ ] Model can save/load weights

#### Browser Demo
- [ ] User can draw digit on canvas (28x28)
- [ ] Real-time classification with <100ms latency
- [ ] Confidence score displayed (0-100%)
- [ ] "Clear" button resets canvas
- [ ] Filter visualization shows learned patterns
- [ ] Activation maps update in real-time
- [ ] Works on mobile (touch input)

### Non-Functional Requirements

#### Performance Targets
- [ ] **Training speed:** 100+ images/sec on batch_size=32
- [ ] **Inference speed:** <10ms per image (single-image forward pass)
- [ ] **Memory:** Stable during long training runs (bounded buffers)
- [ ] **WASM bundle:** CNN crate adds <500 KB to total bundle
- [ ] **Frame rate:** 60 FPS for visualizations

#### Code Quality
- [ ] All public APIs have rustdoc comments with examples
- [ ] Unit tests for each layer (forward, backward, shape calculations)
- [ ] Integration tests for full CNN training
- [ ] Gradient checking tests (numerical vs analytical gradients)
- [ ] Property-based tests for shape invariants
- [ ] Benchmarks for convolution performance
- [ ] Examples run successfully (`cargo run --example mnist_demo -p cnn`)

#### Documentation
- [ ] README.md in cnn/ crate explains architecture
- [ ] CLAUDE.md updated with CNN usage patterns
- [ ] Technical book chapter on CNN implementation (optional)
- [ ] Interactive demo has onboarding tour

### Quality Gates

#### Before PR Merge
- [ ] `cargo test --all` passes (including cnn tests)
- [ ] `cargo clippy --all` has no warnings
- [ ] `cargo fmt --all` applied
- [ ] MNIST demo achieves 95%+ accuracy
- [ ] Browser demo runs at 60 FPS
- [ ] Code reviewed by maintainer
- [ ] All documentation updated

#### Before Release (v0.3.0)
- [ ] Performance benchmarks documented
- [ ] User testing with 5+ people
- [ ] Cross-browser testing (Chrome, Firefox, Safari)
- [ ] Mobile testing (iOS Safari, Chrome Android)
- [ ] Memory leak testing (10+ minute runs)

## Success Metrics

### Technical Metrics
- **Accuracy:** 95%+ on MNIST test set (industry standard for LeNet-5)
- **Training time:** <5 minutes for 10 epochs (on modern laptop)
- **Inference latency:** <10ms (enables real-time digit recognition)
- **Bundle size:** <2 MB total (including MNIST data)
- **Frame rate:** 60 FPS for all visualizations

### User Experience Metrics
- **Time to first prediction:** <10 seconds after page load
- **Drawing responsiveness:** No lag when drawing on canvas
- **Prediction accuracy subjective:** 90%+ of user-drawn digits classified correctly
- **Visualization clarity:** Users can explain what filters detect after 5 minutes

### Educational Metrics
- **Understanding improvement:** 80%+ of users can explain convolution after demo
- **Engagement:** Average session time >5 minutes
- **Sharing:** 30%+ of users share demo with others

## Dependencies & Prerequisites

### Library Dependencies

**New dependencies for `cnn` crate:**
```toml
[dependencies]
linear_algebra = { path = "../linear_algebra" }
neural_network = { path = "../neural_network" }  # Reuse activations, dense layers
rand = "0.8"      # Weight initialization
serde = { version = "1.0", features = ["derive"], optional = true }  # Model serialization

[dev-dependencies]
approx = "0.5"    # Floating-point comparisons
proptest = "1.0"  # Property-based testing
criterion = "0.5" # Benchmarking
```

**New dependencies for `web` crate:**
```toml
[dependencies]
cnn = { path = "../cnn" }
image = { version = "0.25", default-features = false, features = ["png"] }  # MNIST loading
```

### Data Dependencies

**MNIST Dataset:**
- Training images: `train-images-idx3-ubyte` (9.9 MB)
- Training labels: `train-labels-idx1-ubyte` (28 KB)
- Test images: `t10k-images-idx3-ubyte` (1.6 MB)
- Test labels: `t10k-labels-idx1-ubyte` (4.5 KB)

**Total:** ~12 MB (will be embedded in WASM, consider compression or subset)

**Strategy:**
1. Full dataset for CLI training/testing
2. Subset (1000 images) for browser demo to reduce bundle size
3. Load full dataset on-demand via fetch API (progressive enhancement)

### Technical Prerequisites

- [ ] Understand im2col algorithm (read tutorial: https://iq.opengenus.org/im2col/)
- [ ] Review LeNet-5 paper (understand architecture)
- [ ] Study gradient checking technique
- [ ] Learn IDX file format (MNIST data format)

## Risk Analysis & Mitigation

### Risk 1: Performance Below Target (Likelihood: Medium, Impact: High)

**Risk:** Convolution too slow, can't achieve 100 images/sec or 60 FPS.

**Mitigation:**
1. **Benchmark early:** Implement conv2d, measure immediately
2. **Profiling:** Use `cargo flamegraph` to find bottlenecks
3. **Optimization path:**
   - Start with im2col (proven approach)
   - Add zero-allocation `forward_single()` for inference
   - Consider Winograd if still too slow (2.25x speedup for 3x3)
   - Use SIMD if available in stable Rust
4. **Fallback:** Reduce demo resolution (14x14 instead of 28x28)

### Risk 2: Bundle Size Exceeds 2 MB (Likelihood: Medium, Impact: Medium)

**Risk:** MNIST data + CNN code makes WASM too large, slow initial load.

**Mitigation:**
1. **Subset data:** Ship only 1000 MNIST images in browser
2. **Compression:** Use gzip/brotli on WASM bundle (50-70% reduction)
3. **Lazy loading:** Fetch full dataset only when "Train" button clicked
4. **Alternative:** Use synthetic data (circles, lines) for demo instead of MNIST
5. **wasm-opt:** Use `--optimize-level 3` (typically 30% reduction)

### Risk 3: Gradient Checking Fails (Likelihood: Low, Impact: High)

**Risk:** Backpropagation implementation has bugs, model doesn't learn.

**Mitigation:**
1. **Start simple:** Test on tiny dataset (10 images, 2 classes) first
2. **Gradient checking:** Numerical vs analytical gradients (< 1e-5 difference)
3. **Unit tests:** Test each layer independently
4. **Known architecture:** Implement exact LeNet-5 (well-documented, proven)
5. **Debug tools:** Print loss every iteration, visualize weights

### Risk 4: Browser Compatibility Issues (Likelihood: Low, Impact: Medium)

**Risk:** Demo doesn't work in Safari or mobile browsers.

**Mitigation:**
1. **Test early:** Check Safari and mobile in Week 3
2. **Polyfills:** Use Dioxus abstractions (handles browser differences)
3. **Progressive enhancement:** Detect Canvas API, fall back to static images
4. **Touch events:** Test touch input on real devices, not just emulators

## Resource Requirements

### Development Time

**Phase 1: Core CNN Library (2 weeks)**
- Conv2D implementation: 3 days
- Pooling layers: 1 day
- Layer trait + builder: 2 days
- Backpropagation: 3 days
- Testing + debugging: 2 days
- Documentation: 1 day

**Phase 2: MNIST Integration (1 week)**
- Dataset loading: 1 day
- Training loop: 1 day
- Hyperparameter tuning: 2 days
- CLI example: 1 day
- Testing: 1 day
- Documentation: 1 day

**Phase 3: Browser Demo (1 week)**
- Canvas drawing UI: 1 day
- Real-time inference: 1 day
- Filter visualization: 2 days
- Activation maps: 2 days
- Polish + testing: 1 day

**Total:** 4 weeks (160 hours) for full implementation

### Compute Resources

**Training:**
- CPU: Modern laptop (4+ cores) sufficient
- RAM: 4 GB (MNIST fits in memory)
- Storage: 100 MB (code + data)

**Browser:**
- Modern browser (Chrome 90+, Firefox 88+, Safari 14+)
- 2 GB RAM minimum
- GPU optional (not required for MNIST)

### External Dependencies

**Data:**
- MNIST dataset (http://yann.lecun.com/exdb/mnist/)
- License: Public domain (Yann LeCun)

**References:**
- LeNet-5 paper: http://yann.lecun.com/exdb/lenet/
- Im2col tutorial: https://iq.opengenus.org/im2col/
- CS231n CNN notes: https://cs231n.github.io/convolutional-networks/

## Future Considerations

### Extensibility

**After v0.3.0 (MNIST working):**
- [ ] Add BatchNormalization layer
- [ ] Implement Dropout for regularization
- [ ] Support different optimizers (Adam, RMSprop) for CNN training
- [ ] Add data augmentation (rotation, flip, zoom)
- [ ] Implement more architectures (VGG, ResNet blocks)

**After v0.4.0:**
- [ ] CIFAR-10 support (RGB images, 10 classes)
- [ ] Transfer learning (fine-tune pre-trained models)
- [ ] WebGPU backend for GPU acceleration
- [ ] Real-time webcam classification
- [ ] Model export to ONNX format

### Performance Optimizations

**If needed (based on profiling):**
- [ ] Winograd convolution for 3x3 kernels (2.25x speedup)
- [ ] SIMD vectorization (4-8x speedup on certain ops)
- [ ] Quantization (INT8 inference, 4x smaller, 2x faster)
- [ ] Model pruning (remove 70-90% of weights, minimal accuracy loss)

### Alternative Datasets

**Beyond MNIST:**
- Fashion-MNIST (same format, more challenging)
- EMNIST (letters and digits)
- CIFAR-10 (color images, 10 classes)
- Custom datasets (user-uploaded images)

## Documentation Plan

### New Documentation

**Files to create:**
- [ ] `cnn/README.md` - Crate overview, quick start
- [ ] `cnn/ARCHITECTURE.md` - Technical deep dive (im2col, backprop)
- [ ] `docs/tutorials/CNN_TUTORIAL.md` - Step-by-step guide
- [ ] `docs/CNN_VISUALIZATION.md` - How to interpret filters and activations

**Files to update:**
- [ ] `README.md` - Add CNN to "What's New" section
- [ ] `CLAUDE.md` - Update workspace structure, add CNN usage patterns
- [ ] `docs/TECHNICAL_BOOK.md` - Add Chapter 8: Convolutional Neural Networks
- [ ] `web/README.md` - Document CNN demo route

### Code Documentation

**Rustdoc requirements:**
- [ ] Module-level docs for `cnn` crate (//!)
- [ ] Every public struct/enum/function has /// doc comments
- [ ] Examples in doc comments (tested via doc tests)
- [ ] Links to related items using [ItemName]

**Example:**
```rust
/// Convolutional layer for feature extraction from images
///
/// Applies 2D convolution using the im2col algorithm for efficient
/// matrix multiplication. Supports configurable kernel size, stride,
/// and padding.
///
/// # Algorithm
///
/// 1. **im2col:** Rearrange input patches into columns
/// 2. **Matrix multiply:** `weights × columns`
/// 3. **col2im:** Reshape result to output dimensions
///
/// # Examples
///
/// ```
/// use cnn::layers::Conv2D;
/// use linear_algebra::matrix::Matrix;
///
/// let mut conv = Conv2D::new(6, (5, 5), (1, 1), (0, 0))?;
/// let input = Matrix::zeros(28, 28); // 28x28 grayscale image
/// let output = conv.forward(&input)?; // 24x24x6 feature maps
/// ```
///
/// # Performance
///
/// - **Forward:** O(K² × C_in × H_out × W_out × C_out)
/// - **Memory:** O(K² × C_in × H_out × W_out) for im2col cache
///
/// where K=kernel_size, C_in=input_channels, C_out=output_channels
///
/// # See Also
///
/// - [Pooling2D] for dimensionality reduction
/// - [Flatten] for transition to dense layers
pub struct Conv2D { ... }
```

## References & Research

### Internal References

**Similar patterns in codebase:**
- `neural_network/src/lib.rs:1-547` - Layer abstraction, training loop pattern
- `neural_network/src/optimizer.rs:536-601` - Zero-allocation specialization (apply to `forward_single()`)
- `web/src/components/optimizer_demo.rs` - Interactive visualization component structure
- `linear_algebra/src/matrix.rs:201-232` - Matrix multiplication (reuse for im2col)

**Architecture decisions:**
- `CLAUDE.md:46-57` - Zero-allocation hot paths, bounded memory, standard indexing
- `docs/TECHNICAL_BOOK.md` - Chapter 3 (zero-allocation), Chapter 5 (browser performance)

### External References

**Academic Papers:**
- LeNet-5: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf (Original CNN architecture)
- Im2col: https://hal.inria.fr/inria-00112631/document (Efficient convolution)
- Winograd: https://arxiv.org/abs/1509.09308 (Fast CNN algorithms)
- Grad-CAM: https://arxiv.org/abs/1610.02391 (Visualization technique)

**Tutorials & Documentation:**
- CS231n CNN notes: https://cs231n.github.io/convolutional-networks/
- Im2col explained: https://iq.opengenus.org/im2col/
- Gradient checking: https://cs231n.github.io/neural-networks-3/#gradcheck
- MNIST format: http://yann.lecun.com/exdb/mnist/

**Rust Libraries:**
- ndarray: https://docs.rs/ndarray/latest/ndarray/ (Array operations reference)
- image-rs: https://docs.rs/image/latest/image/ (Image loading for MNIST)
- Burn CNN: https://burn.dev/book/building-blocks/backend.html (Reference implementation)

**Browser ML:**
- WebGPU: https://webgpufundamentals.org (Future GPU acceleration)
- WASM SIMD: https://v8.dev/features/simd (Potential optimization)

### Related Work

**Similar projects:**
- TensorFlow.js: https://www.tensorflow.org/js (Pure JS, slower than WASM)
- ONNX Runtime Web: https://onnxruntime.ai/docs/tutorials/web/ (Inference-only)
- ML5.js: https://ml5js.org (Educational, pre-trained models)

**Differentiators:**
- ✅ Train models in browser (not just inference)
- ✅ Interactive visualization of learning process
- ✅ Rust performance (10-50x faster than JS)
- ✅ Zero-backend architecture
- ✅ Educational focus (understand how CNNs work)

---

## Implementation Checklist

### Pre-Implementation
- [ ] Read LeNet-5 paper
- [ ] Study im2col algorithm
- [ ] Review gradient checking technique
- [ ] Download MNIST dataset
- [ ] Set up benchmarking harness

### Phase 1: Core Library
- [ ] Create `cnn` crate with proper structure
- [ ] Implement Conv2D layer (forward + backward)
- [ ] Implement Pooling2D (Max and Average)
- [ ] Implement Flatten layer
- [ ] Add Layer trait
- [ ] Build CNN struct with builder pattern
- [ ] Write unit tests (gradient checking!)
- [ ] Add benchmarks
- [ ] Document all public APIs

### Phase 2: MNIST
- [ ] Add MNIST loading code
- [ ] Create CLI training example
- [ ] Verify 95%+ accuracy
- [ ] Profile performance
- [ ] Optimize if needed
- [ ] Document training process

### Phase 3: Browser Demo
- [ ] Create `cnn_demo.rs` component
- [ ] Implement canvas drawing
- [ ] Add real-time inference
- [ ] Build filter visualization
- [ ] Build activation visualization
- [ ] Add mobile touch support
- [ ] Cross-browser testing
- [ ] Performance profiling

### Finalization
- [ ] Update all documentation
- [ ] Create PR with comprehensive description
- [ ] Code review
- [ ] User testing
- [ ] Merge and release v0.3.0

---

**Estimated Total Effort:** 160 hours (4 weeks full-time)
**Target Release:** v0.3.0
**Priority:** 🔥 HIGHEST - Unlock the killer app for browser ML

---

**Files referenced in this issue:**
- `/Users/brunodossantos/Code/brunoml/cargo_workspace/neural_network/src/lib.rs`
- `/Users/brunodossantos/Code/brunoml/cargo_workspace/neural_network/src/optimizer.rs:536-601`
- `/Users/brunodossantos/Code/brunoml/cargo_workspace/web/src/components/optimizer_demo.rs`
- `/Users/brunodossantos/Code/brunoml/cargo_workspace/linear_algebra/src/matrix.rs:201-232`
- `/Users/brunodossantos/Code/brunoml/cargo_workspace/CLAUDE.md`
- `/Users/brunodossantos/Code/brunoml/cargo_workspace/docs/TECHNICAL_BOOK.md`
