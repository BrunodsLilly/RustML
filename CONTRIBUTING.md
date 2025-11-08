# Contributing to the ML+Rust+WASM Project

Welcome! This guide will help you contribute effectively to this revolutionary client-side ML platform. Whether you're working on core libraries, web components, or documentation, this guide covers everything you need to know.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Development Philosophy](#development-philosophy)
3. [TDD Workflow](#tdd-workflow)
4. [Library Development](#library-development)
5. [Dioxus Component Development](#dioxus-component-development)
6. [Trait System Best Practices](#trait-system-best-practices)
7. [Testing Strategy](#testing-strategy)
8. [Performance Guidelines](#performance-guidelines)
9. [Code Review Process](#code-review-process)
10. [Getting Help](#getting-help)

---

## Quick Start

### Prerequisites

```bash
# Install Rust (latest stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Dioxus CLI
cargo install dioxus-cli

# Install Playwright for E2E tests (Dioxus web components)
cd web && npm install
```

### First Contribution Workflow

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/cargo_workspace.git
cd cargo_workspace

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Run tests to ensure clean starting state
cargo test --all

# 4. Make your changes using TDD (see below)

# 5. Run tests again
cargo test --all

# 6. Run E2E tests for web changes
cd web && npx playwright test

# 7. Commit and push
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name

# 8. Create a pull request on GitHub
```

---

## Development Philosophy

### Revolutionary Goals

This project showcases what's possible when **Rust + WASM meet machine learning**:

1. **Zero-backend computation** - Everything runs in the browser
2. **Native performance** - 1000+ optimizer iterations/sec, 60 FPS rendering
3. **Educational excellence** - Teach ML concepts through interaction
4. **Production quality** - Professional error handling, accessibility, polish

### Core Principles

1. **Performance First** - Profile, don't guess. Specialize for common cases.
2. **Test-Driven** - Write tests before implementation. Tests document behavior.
3. **Type Safety** - Use Rust's type system to prevent bugs at compile time.
4. **Educational Value** - Every feature should teach something about ML or Rust.

---

## TDD Workflow

### The Red-Green-Refactor Cycle

Test-Driven Development (TDD) is **mandatory** for this project. Here's the workflow:

```
┌──────────────────────────────────────┐
│  1. RED: Write a failing test       │
│     (Defines the API and behavior)   │
└─────────────┬────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│  2. GREEN: Write minimal code       │
│     to make test pass                │
└─────────────┬────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│  3. REFACTOR: Clean up code         │
│     (Tests still pass)               │
└─────────────┬────────────────────────┘
              │
              ▼
              └──────► Repeat
```

### TDD Best Practices (2025)

Based on current industry standards:

1. **Start Small** - Write tests for small pieces of functionality
2. **Test Independence** - Tests should not rely on external conditions or other tests
3. **Run in Any Order** - Use mock objects and test doubles for isolation
4. **One Assertion Focus** - Each test should verify one specific behavior
5. **Clear Test Names** - Test name should describe what behavior is being tested

### Example: TDD for a New Optimizer

Let's walk through adding a new optimizer (NAG - Nesterov Accelerated Gradient) using TDD.

#### Step 1: RED - Write the Test First

Create `neural_network/tests/optimizer_tests.rs` (or add to existing):

```rust
#[test]
fn test_nag_creation() {
    // Define the API we want
    let nag = Optimizer::nag(0.01, 0.9);

    // Verify creation succeeds
    assert_eq!(nag.optimizer_type(), OptimizerType::NAG);
    assert_eq!(nag.learning_rate(), 0.01);
    assert_eq!(nag.momentum(), 0.9);
}
```

**Run the test:**
```bash
cargo test -p neural_network test_nag_creation
```

**Expected:** ❌ Compilation error - `OptimizerType::NAG` doesn't exist.

#### Step 2: GREEN - Minimal Implementation

Add to `neural_network/src/optimizer.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizerType {
    SGD,
    Momentum,
    RMSprop,
    Adam,
    NAG,  // New variant
}

impl Optimizer {
    pub fn nag(learning_rate: f64, momentum: f64) -> Self {
        assert!(learning_rate > 0.0 && learning_rate.is_finite(),
            "Learning rate must be positive and finite");
        assert!(momentum >= 0.0 && momentum < 1.0,
            "Momentum must be in [0, 1)");

        Optimizer {
            optimizer_type: OptimizerType::NAG,
            learning_rate,
            beta1: momentum,  // Reuse existing field
            beta2: 0.0,
            epsilon: 0.0,
            t: 0,
            velocity: HashMap::new(),
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn momentum(&self) -> f64 {
        self.beta1
    }
}
```

**Run the test:**
```bash
cargo test -p neural_network test_nag_creation
```

**Expected:** ✅ Test passes!

#### Step 3: Add Behavior Test (RED again)

```rust
#[test]
fn test_nag_lookahead() {
    // NAG uses "lookahead" - updates velocity with future gradient
    let mut opt = Optimizer::nag(0.1, 0.9);
    let mut weights = mat(vec![1.0], 1, 1);
    let gradient = mat(vec![1.0], 1, 1);
    let layer_shapes = vec![(1, 1)];

    // First step
    opt.update_weights(0, &gradient, &mut weights, &layer_shapes);
    let step1 = 1.0 - weights[(0, 0)];

    // NAG should take a larger step than plain momentum on first iteration
    // because it uses the "lookahead" gradient
    assert!(step1 > 0.1, "NAG should take >0.1 step with lookahead");
}
```

**Run:** ❌ Fails - NAG not implemented in `update_weights`.

#### Step 4: Implement NAG Logic (GREEN)

```rust
impl Optimizer {
    pub fn update_weights(/* ... */) {
        match self.optimizer_type {
            // ... existing cases
            OptimizerType::NAG => {
                // NAG: v_{t+1} = μ * v_t + ∇f(w_t + μ * v_t)
                // w_{t+1} = w_t - lr * v_{t+1}

                let velocity_key = format!("w_{}", layer_idx);
                let velocity = self.velocity.entry(velocity_key.clone())
                    .or_insert_with(|| Matrix::zeros(weights.rows, weights.cols));

                // Lookahead position: w + μ * v
                let lookahead = weights.clone() + velocity.clone() * self.beta1;

                // Update velocity with gradient at lookahead
                *velocity = velocity.clone() * self.beta1 + gradient.clone();

                // Update weights
                for i in 0..weights.rows {
                    for j in 0..weights.cols {
                        weights[(i, j)] -= self.learning_rate * velocity[(i, j)];
                    }
                }
            }
        }
    }
}
```

**Run:** ✅ Test passes!

#### Step 5: REFACTOR

Now that tests pass, clean up:

```rust
// Extract common velocity update pattern
impl Optimizer {
    fn get_or_init_velocity(&mut self, key: String, rows: usize, cols: usize) -> &mut Matrix<f64> {
        self.velocity.entry(key)
            .or_insert_with(|| Matrix::zeros(rows, cols))
    }
}
```

**Run tests again:** ✅ Still passing after refactor!

#### Key TDD Lessons from This Example

1. **Tests define the API** - We decided on `Optimizer::nag(lr, momentum)` in the test
2. **Small steps** - Created enum variant before implementing logic
3. **Tests document behavior** - Test name and assertions explain NAG's "lookahead"
4. **Refactor safely** - Once tests pass, we can refactor with confidence
5. **Regression protection** - Future changes won't break NAG without failing tests

---

## Library Development

### Package Structure

```
cargo_workspace/
├── linear_algebra/      ← Core math primitives
│   ├── src/
│   │   ├── lib.rs
│   │   ├── matrix.rs    ← Generic Matrix<T> with operator overloading
│   │   └── vectors.rs   ← Generic Vector<T>
│   └── tests/
│       └── integration_tests.rs
│
├── neural_network/      ← ML algorithms
│   ├── src/
│   │   ├── lib.rs
│   │   ├── activation.rs
│   │   └── optimizer.rs ← SGD, Momentum, RMSprop, Adam
│   ├── tests/
│   │   └── optimizer_tests.rs  ← Comprehensive TDD tests
│   └── examples/
│       └── xor_demo.rs
│
└── linear_regression/   ← Simpler ML example
    └── src/lib.rs
```

### Adding a New Feature to a Library

#### Workflow

1. **Write tests first** (TDD red phase)
2. **Implement minimal solution** (TDD green phase)
3. **Refactor for clarity** (TDD refactor phase)
4. **Add documentation** (doc comments with examples)
5. **Add to public API** if appropriate
6. **Run full test suite** (`cargo test -p your_package`)

#### Example: Adding Matrix Determinant

**File:** `linear_algebra/src/matrix.rs`

**Step 1: Test (in `linear_algebra/tests/integration_tests.rs` or inline `#[cfg(test)]`):**

```rust
#[test]
fn test_determinant_2x2() {
    let m = Matrix::from_vec(vec![3.0, 8.0, 4.0, 6.0], 2, 2).unwrap();
    // det = 3*6 - 8*4 = 18 - 32 = -14
    assert!((m.determinant().unwrap() - (-14.0)).abs() < 1e-10);
}

#[test]
fn test_determinant_non_square() {
    let m = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
    assert!(m.determinant().is_err(), "Non-square matrix has no determinant");
}
```

**Step 2: Implementation:**

```rust
impl<T> Matrix<T>
where
    T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// Compute the determinant of a square matrix
    pub fn determinant(&self) -> Result<T, String> {
        if self.rows != self.cols {
            return Err(format!(
                "Cannot compute determinant of non-square matrix ({}x{})",
                self.rows, self.cols
            ));
        }

        match self.rows {
            0 => Ok(T::default()),
            1 => Ok(self[(0, 0)]),
            2 => {
                // det(2x2) = ad - bc
                Ok(self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)])
            }
            _ => {
                // For n>2, use cofactor expansion (not optimal, but clear)
                // TODO: Implement LU decomposition for better performance
                unimplemented!("Determinant for matrices > 2x2 not yet implemented")
            }
        }
    }
}
```

**Step 3: Documentation:**

```rust
/// Compute the determinant of a square matrix.
///
/// # Examples
///
/// ```
/// use linear_algebra::matrix::Matrix;
///
/// let m = Matrix::from_vec(vec![3.0, 8.0, 4.0, 6.0], 2, 2).unwrap();
/// assert_eq!(m.determinant().unwrap(), -14.0);
/// ```
///
/// # Errors
///
/// Returns `Err` if the matrix is not square.
///
/// # Performance
///
/// Current implementation is O(n!) for n>2 matrices. For production use,
/// consider implementing LU decomposition (O(n³)).
pub fn determinant(&self) -> Result<T, String> {
    // ... implementation
}
```

**Step 4: Verify:**

```bash
cargo test -p linear_algebra
cargo doc --open  # Check documentation renders correctly
```

### Trait-Based Design

See [`docs/TRAIT_SYSTEM_RESEARCH.md`](docs/TRAIT_SYSTEM_RESEARCH.md) for comprehensive guidance.

**Quick checklist for new traits:**

- [ ] Does it solve a real problem in the codebase?
- [ ] Is it focused on a single responsibility?
- [ ] Are the trait bounds minimal (only what's needed)?
- [ ] Does it provide default implementations where sensible?
- [ ] Is it documented with examples?
- [ ] Have you implemented it for relevant types?
- [ ] Do tests cover the trait's contract?

---

## Dioxus Component Development

### Dioxus Component Structure

Dioxus is a React-like framework for Rust. Components are functions that return RSX (Rust JSX).

**Basic Component Pattern:**

```rust
use dioxus::prelude::*;

#[component]
pub fn MyComponent() -> Element {
    // State management
    let mut count = use_signal(|| 0);

    // Render UI
    rsx! {
        div { class: "my-component",
            h2 { "Count: {count}" }
            button {
                onclick: move |_| count += 1,
                "Increment"
            }
        }
    }
}
```

### TDD for Dioxus Components

Dioxus components are tested using **End-to-End (E2E) tests** with Playwright.

#### Test Structure

**Location:** `web/tests/*.spec.js`

**Pattern:**

```javascript
const { test, expect } = require('@playwright/test');

test.describe('Component Name', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:8080/route');
  });

  test('should display initial state', async ({ page }) => {
    // Verify initial render
    await expect(page.getByRole('heading', { name: 'Count: 0' })).toBeVisible();
  });

  test('should update when button clicked', async ({ page }) => {
    // Interact with component
    await page.getByRole('button', { name: 'Increment' }).click();

    // Verify state change
    await expect(page.getByRole('heading', { name: 'Count: 1' })).toBeVisible();
  });
});
```

### TDD Workflow for Dioxus Components

#### Example: Adding a "Reset" Button

**Step 1: RED - Write Failing Test**

```javascript
// web/tests/showcase.spec.js
test('should reset count to zero', async ({ page }) => {
  await page.goto('http://localhost:8080/showcase');

  // Increment count
  await page.getByRole('button', { name: 'Increment' }).click();
  await expect(page.getByRole('heading', { name: 'Count: 1' })).toBeVisible();

  // Reset
  await page.getByRole('button', { name: 'Reset' }).click();

  // Verify reset to zero
  await expect(page.getByRole('heading', { name: 'Count: 0' })).toBeVisible();
});
```

**Run:** ❌ Fails - "Reset" button doesn't exist

**Step 2: GREEN - Implement Feature**

```rust
// web/src/components/showcase.rs
#[component]
pub fn Counter() -> Element {
    let mut count = use_signal(|| 0);

    rsx! {
        div { class: "counter",
            h2 { "Count: {count}" }
            button {
                onclick: move |_| count += 1,
                "Increment"
            }
            button {
                onclick: move |_| count.set(0),
                "Reset"
            }
        }
    }
}
```

**Step 3: Run Dev Server and Test**

```bash
# Terminal 1: Start dev server
cd web && dx serve

# Terminal 2: Run tests
npx playwright test --grep "should reset count"
```

**Expected:** ✅ Test passes!

### Dioxus Testing Best Practices

1. **Use Semantic Queries**
   ```javascript
   // GOOD: Semantic role-based queries
   page.getByRole('button', { name: 'Submit' })
   page.getByRole('heading', { level: 1 })

   // BAD: Fragile CSS selectors
   page.locator('.btn-primary')
   ```

2. **Wait for State Changes**
   ```javascript
   // GOOD: Wait for expected state
   await page.getByText('Loading...').waitFor({ state: 'hidden' });
   await expect(page.getByText('Data loaded')).toBeVisible();

   // BAD: Arbitrary timeouts
   await page.waitForTimeout(1000);
   ```

3. **Test User Workflows, Not Implementation**
   ```javascript
   // GOOD: Test what user experiences
   test('user can submit form', async ({ page }) => {
     await page.fill('input[name="email"]', 'user@example.com');
     await page.click('button[type="submit"]');
     await expect(page.getByText('Success!')).toBeVisible();
   });

   // BAD: Testing internal state
   test('form state updates', async ({ page }) => {
     // Don't test Dioxus signal internals
   });
   ```

4. **Isolate Tests**
   ```javascript
   test.beforeEach(async ({ page }) => {
     // Fresh page for each test
     await page.goto('http://localhost:8080/showcase');
   });
   ```

### Performance Testing for Dioxus

For performance-critical components (like the optimizer visualizer), add manual performance tests:

```javascript
test('optimizer should handle 1000+ iterations/sec', async ({ page }) => {
  await page.goto('http://localhost:8080/optimizers');

  // Start optimizer
  await page.click('button[name="Start"]');

  // Wait 5 seconds
  await page.waitForTimeout(5000);

  // Get iteration count
  const iterations = await page.locator('.iteration-count').textContent();
  const count = parseInt(iterations);

  // Verify > 5000 iterations (1000/sec minimum)
  expect(count).toBeGreaterThan(5000);
});
```

---

## Trait System Best Practices

### Quick Reference

See full guide: [`docs/TRAIT_SYSTEM_RESEARCH.md`](docs/TRAIT_SYSTEM_RESEARCH.md)

**Key Principles:**

1. **Start Simple** - Focused traits with single responsibility
2. **Progressive Bounds** - Add constraints only when needed
3. **Operator Overloading** - Implement `Add`, `Mul`, etc. for natural syntax
4. **Zero-Cost** - Traits compile to monomorphized code (no runtime cost)
5. **Document Bounds** - Explain why each trait bound is necessary

**Common Patterns in This Codebase:**

```rust
// Pattern 1: Generic numeric operations
impl<T> Matrix<T>
where
    T: Copy + Default + Add<Output = T>,
{
    pub fn add(self, other: Self) -> Self { /* ... */ }
}

// Pattern 2: Progressive capability exposure
impl<T: Copy + Default> Matrix<T> {
    pub fn new(rows: usize, cols: usize) -> Self { /* ... */ }
}

impl<T: Copy + Default + From<i32>> Matrix<T> {
    pub fn zeros(rows: usize, cols: usize) -> Self { /* ... */ }
    pub fn ones(rows: usize, cols: usize) -> Self { /* ... */ }
}

// Pattern 3: Operator overloading for natural syntax
impl<T> Add for Matrix<T>
where
    T: Copy + Default + Add<Output = T>,
{
    type Output = Self;
    fn add(self, other: Self) -> Self::Output { /* ... */ }
}
```

---

## Testing Strategy

### Testing Pyramid

```
         ╱╲
        ╱E2E╲         ← Few, critical user workflows
       ╱──────╲
      ╱Integ.  ╲      ← Integration tests (components working together)
     ╱──────────╲
    ╱  Unit Tests╲    ← Many, fast, isolated unit tests
   ╱──────────────╲
```

### Unit Tests (Rust Libraries)

**Location:** Inline with code or in `tests/` directory

**Example:** `neural_network/tests/optimizer_tests.rs`

**Run:**
```bash
cargo test -p neural_network
cargo test -p linear_algebra
cargo test --all  # All packages
```

**Best Practices:**
- Test one behavior per test
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Test edge cases and error conditions

**Example Structure:**
```rust
#[test]
fn test_optimizer_invalid_learning_rate() {
    // Arrange: Set up invalid input
    let invalid_lr = -0.01;

    // Act & Assert: Verify panic with expected message
    let result = std::panic::catch_unwind(|| {
        Optimizer::sgd(invalid_lr)
    });

    assert!(result.is_err(), "Should panic on negative learning rate");
}
```

### Integration Tests (Rust Libraries)

**Location:** `<package>/tests/*.rs`

**Purpose:** Test how multiple modules work together

**Example:**
```rust
// linear_algebra/tests/integration_tests.rs
use linear_algebra::{matrix::Matrix, vectors::Vector};

#[test]
fn test_matrix_vector_multiplication_end_to_end() {
    // Create matrix and vector
    let m = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let v = Vector { data: vec![5.0, 6.0] };

    // Matrix-vector multiplication
    let result = m * v;

    // Verify result
    assert_eq!(result.data, vec![17.0, 39.0]);
}
```

### E2E Tests (Dioxus Components)

**Location:** `web/tests/*.spec.js`

**Run:**
```bash
cd web
dx serve  # Terminal 1
npx playwright test  # Terminal 2
```

**Example:** See `web/tests/showcase.spec.js` for comprehensive examples

---

## Performance Guidelines

### General Principles

1. **Profile First** - Use `cargo bench` or browser DevTools before optimizing
2. **Hot Path Optimization** - Focus on code that runs frequently
3. **Zero-Allocation Patterns** - Avoid allocations in tight loops
4. **Specialize for Common Cases** - Example: `Optimizer::step_2d()` uses scalars instead of `Matrix<T>`

### Example: Zero-Allocation Optimization

**Before (24,000 allocations/sec):**
```rust
fn optimizer_step(x: f64, y: f64, dx: f64, dy: f64) -> (f64, f64) {
    let mut weights = Matrix::from_vec(vec![x, y], 1, 2)?;  // Allocation!
    let gradient = Matrix::from_vec(vec![dx, dy], 1, 2)?;  // Allocation!
    optimizer.update_weights(0, &gradient, &mut weights, &shapes);
    (weights[(0, 0)], weights[(0, 1)])
}
```

**After (0 allocations, 10-50x faster):**
```rust
fn optimizer_step(&mut self, x: f64, y: f64, dx: f64, dy: f64) -> (f64, f64) {
    // Pure scalar math, no heap allocations
    self.step_2d((x, y), (dx, dy))
}
```

**Lesson:** For hot paths with known dimensions, specialize instead of using generic code.

### Performance Checklist

- [ ] Profile before optimizing (`cargo bench`, Chrome DevTools)
- [ ] Identify hot paths (functions called in tight loops)
- [ ] Consider zero-allocation alternatives for hot paths
- [ ] Use `#[inline]` for small, frequently-called functions
- [ ] Prefer compile-time generics over runtime trait objects
- [ ] Benchmark after changes to verify improvement

See: `docs/PERFORMANCE_BENCHMARK.md` for detailed benchmarking guide.

---

## Code Review Process

### Before Submitting

- [ ] All tests pass (`cargo test --all`)
- [ ] E2E tests pass (if web changes: `npx playwright test`)
- [ ] Code is formatted (`cargo fmt`)
- [ ] No compiler warnings (`cargo clippy -- -D warnings`)
- [ ] Documentation is updated
- [ ] Commit messages follow conventional commits format
- [ ] Performance impact assessed (if applicable)

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Performance improvement
- [ ] Documentation
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] E2E tests added/updated (for web changes)
- [ ] Manual testing performed

## Performance Impact
Describe any performance implications (positive or negative)

## Screenshots (if applicable)
For UI changes, include before/after screenshots

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Criteria

Reviewers will check:

1. **Correctness** - Does it work as intended?
2. **Tests** - Are there comprehensive tests?
3. **Performance** - Any negative performance impact?
4. **Type Safety** - Are trait bounds appropriate?
5. **Documentation** - Is it clear how to use this?
6. **Simplicity** - Is this the simplest solution?

---

## Getting Help

### Documentation Resources

- **Project Docs:** `docs/` directory
  - [`CLAUDE.md`](CLAUDE.md) - Project architecture and workflow
  - [`docs/TRAIT_SYSTEM_RESEARCH.md`](docs/TRAIT_SYSTEM_RESEARCH.md) - Trait system deep dive
  - [`docs/PERFORMANCE_BENCHMARK.md`](docs/PERFORMANCE_BENCHMARK.md) - How to benchmark
  - [`docs/TECHNICAL_BOOK.md`](docs/TECHNICAL_BOOK.md) - Comprehensive technical guide

- **Rust Resources:**
  - [The Rust Book](https://doc.rust-lang.org/book/)
  - [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
  - [Rust Design Patterns](https://rust-unofficial.github.io/patterns/)

- **Dioxus Resources:**
  - [Dioxus Documentation](https://dioxuslabs.com/learn/0.7/)
  - [Dioxus GitHub](https://github.com/DioxusLabs/dioxus)
  - [Dioxus Examples](https://github.com/DioxusLabs/dioxus/tree/main/examples)

### Common Issues

#### "Tests fail with panic in WASM"

**Solution:** WASM panics are silent. Use `console::error_1()` for debugging:

```rust
use web_sys::console;

if !condition {
    console::error_1(&"Descriptive error message".into());
    return;
}
```

#### "Matrix dimensions mismatch at runtime"

**Solution:** This is a known limitation. Consider:

1. Adding dimension checks with clear error messages
2. Using const generics for compile-time dimension checking (future improvement)

```rust
impl<T> Matrix<T> {
    pub fn multiply(&self, other: &Matrix<T>) -> Result<Matrix<T>, String> {
        if self.cols != other.rows {
            return Err(format!(
                "Cannot multiply {}x{} matrix with {}x{} matrix. \
                 Inner dimensions must match (cols of first = rows of second).",
                self.rows, self.cols, other.rows, other.cols
            ));
        }
        // ... multiplication logic
    }
}
```

#### "Dioxus hot reload not working"

**Solution:**

```bash
# Ensure using --hot-reload flag
dx serve --hot-reload

# If still not working, kill and restart
killall dx
dx serve --hot-reload
```

### Getting Support

1. **Read the docs first** - Most answers are in `docs/` or `CLAUDE.md`
2. **Search existing issues** - Problem might already be solved
3. **Ask in discussions** - For questions and brainstorming
4. **File an issue** - For bugs or feature requests

---

## Appendix: Example TDD Session

### Scenario: Add Element-Wise Matrix Multiplication

**Goal:** Implement Hadamard product (element-wise multiplication) for matrices.

#### Step 1: RED - Write the Test

```rust
// linear_algebra/tests/integration_tests.rs
#[test]
fn test_matrix_hadamard_product() {
    let m1 = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let m2 = Matrix::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();

    let result = m1.hadamard(&m2);

    // Element-wise: [1*5, 2*6, 3*7, 4*8] = [5, 12, 21, 32]
    assert_eq!(result[(0, 0)], 5.0);
    assert_eq!(result[(0, 1)], 12.0);
    assert_eq!(result[(1, 0)], 21.0);
    assert_eq!(result[(1, 1)], 32.0);
}

#[test]
fn test_hadamard_dimension_mismatch() {
    let m1 = Matrix::from_vec(vec![1.0, 2.0], 1, 2).unwrap();
    let m2 = Matrix::from_vec(vec![1.0, 2.0, 3.0], 1, 3).unwrap();

    let result = std::panic::catch_unwind(|| {
        m1.hadamard(&m2)
    });

    assert!(result.is_err(), "Should panic on dimension mismatch");
}
```

**Run:** `cargo test -p linear_algebra test_matrix_hadamard`

**Expected:** ❌ Compilation error - `hadamard` method doesn't exist.

#### Step 2: GREEN - Minimal Implementation

```rust
// linear_algebra/src/matrix.rs
impl<T> Matrix<T>
where
    T: Copy + Default + Mul<Output = T>,
{
    /// Compute element-wise (Hadamard) product of two matrices
    pub fn hadamard(&self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.shape(), other.shape(),
            "Hadamard product requires same dimensions");

        let data: Vec<T> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}
```

**Run:** `cargo test -p linear_algebra test_matrix_hadamard`

**Expected:** ✅ Tests pass!

#### Step 3: REFACTOR - Improve Error Handling

```rust
/// Compute element-wise (Hadamard) product of two matrices.
///
/// # Examples
///
/// ```
/// use linear_algebra::matrix::Matrix;
///
/// let m1 = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
/// let m2 = Matrix::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
/// let result = m1.hadamard(&m2);
/// assert_eq!(result[(0, 0)], 5.0);  // 1 * 5
/// ```
///
/// # Panics
///
/// Panics if matrix dimensions don't match.
pub fn hadamard(&self, other: &Matrix<T>) -> Matrix<T> {
    assert_eq!(
        self.shape(),
        other.shape(),
        "Hadamard product requires matrices of the same shape. \
         Got {}x{} and {}x{}.",
        self.rows, self.cols, other.rows, other.cols
    );

    let data: Vec<T> = self.data.iter()
        .zip(other.data.iter())
        .map(|(&a, &b)| a * b)
        .collect();

    Matrix {
        data,
        rows: self.rows,
        cols: self.cols,
    }
}
```

**Run:** `cargo test -p linear_algebra`

**Expected:** ✅ All tests still pass!

#### Step 4: Documentation and Examples

```bash
# Generate docs and verify examples compile
cargo doc --open

# Check that doc example runs correctly
cargo test --doc -p linear_algebra
```

**Commit:**
```bash
git add linear_algebra/src/matrix.rs linear_algebra/tests/integration_tests.rs
git commit -m "feat: add Hadamard (element-wise) product for matrices

- Implement Matrix::hadamard() method
- Add comprehensive tests for correct behavior and error cases
- Document with examples and panic conditions
- All tests passing

Closes #123"
```

---

## Summary

Contributing to this project is about:

1. **TDD mindset** - Tests first, implementation second
2. **Type safety** - Let Rust's type system catch bugs
3. **Performance awareness** - Profile and optimize hot paths
4. **Educational value** - Make code clear and well-documented

**Remember:** Every contribution makes this project more revolutionary. Whether you're adding a new optimizer, improving documentation, or fixing a bug, you're helping prove that **Rust + WASM can deliver native ML performance in the browser**.

**Welcome to the revolution!** 🚀

---

**Last Updated:** November 2025
**Questions?** See [`docs/`](docs/) or file an issue on GitHub.
