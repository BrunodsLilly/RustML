# feat: Implement K-Means Clustering with Interactive Visualization

## Overview

Add unsupervised learning capabilities through K-Means clustering with a highly interactive browser-based visualization. This demonstrates a different ML paradigm (unsupervised vs supervised) and provides one of the most visually intuitive learning experiences possible.

**Priority:** 🟢 MEDIUM-HIGH - Different paradigm, extremely visual, moderate implementation complexity (2-3 weeks).

## Problem Statement

### Current Limitations
- Only supervised learning supported (classification, regression)
- No clustering/unsupervised methods
- Cannot demonstrate data exploration without labels
- Missing foundational unsupervised algorithm

### Why This Matters
1. **Different Paradigm:** Introduces unsupervised learning (exploratory data analysis)
2. **Visual Learning:** K-Means is *the* algorithm for teaching clustering concepts
3. **Real-Time Animation:** Watch centroids move iteration-by-iteration (60 FPS)
4. **Interactive Exploration:** Users create data points, choose K, see results instantly
5. **Production Relevance:** K-Means used for customer segmentation, image compression, anomaly detection

### User Stories
- **As a student:** I want to click to add points and watch K-Means group them into clusters
- **As a data scientist:** I want to explore datasets visually before applying supervised learning
- **As an educator:** I want to show students how clustering discovers patterns without labels
- **As a developer:** I want to segment users based on behavior patterns (client-side, privacy-preserving)

## Proposed Solution

### High-Level Architecture

```
New Crate: clustering/
├─ src/
│  ├─ lib.rs              # KMeans struct, core algorithm
│  ├─ kmeans.rs           # Main implementation
│  ├─ initialization.rs   # K-Means++, Random, Manual
│  └─ metrics.rs          # Silhouette score, elbow method
├─ examples/
│  ├─ kmeans_demo.rs      # CLI example on 2D data
│  └─ iris_clustering.rs  # Classic Iris dataset
├─ tests/
│  └─ kmeans_tests.rs     # Convergence, initialization tests
└─ benches/
   └─ clustering_bench.rs # Performance benchmarking
```

### Algorithm Implementation

**K-Means Algorithm:**
```
1. Initialize K centroids (random, K-Means++, or manual)
2. Repeat until convergence:
   a. Assign each point to nearest centroid (Euclidean distance)
   b. Update centroids to mean of assigned points
   c. Check convergence (centroids stopped moving)
3. Return final centroids and assignments
```

**Code Structure:**
```rust
// clustering/src/lib.rs

use linear_algebra::vectors::Vector;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitializationMethod {
    Random,        // Random points from dataset
    KMeansPlusPlus, // Smart initialization (better convergence)
    Manual,         // User-specified initial centroids
}

pub struct KMeans {
    k: usize,                        // Number of clusters
    max_iterations: usize,           // Prevent infinite loops
    tolerance: f64,                  // Convergence threshold
    initialization: InitializationMethod,

    // State (updated during fitting)
    centroids: Vec<Vector<f64>>,     // Current cluster centers
    assignments: Vec<usize>,         // Point → cluster mapping
    inertia: f64,                    // Sum of squared distances (quality metric)
    iteration_count: usize,
}

impl KMeans {
    /// Create a K-Means clusterer
    ///
    /// # Arguments
    /// * `k` - Number of clusters (must be > 0 and < n_samples)
    /// * `max_iterations` - Maximum iterations before stopping
    /// * `tolerance` - Convergence threshold (stop if centroids move < tolerance)
    ///
    /// # Examples
    /// ```
    /// use clustering::KMeans;
    ///
    /// let kmeans = KMeans::new(3, 100, 1e-4);
    /// ```
    pub fn new(k: usize, max_iterations: usize, tolerance: f64) -> Self {
        assert!(k > 0, "K must be positive, got: {}", k);
        assert!(tolerance > 0.0, "Tolerance must be positive, got: {}", tolerance);

        Self {
            k,
            max_iterations,
            tolerance,
            initialization: InitializationMethod::KMeansPlusPlus,
            centroids: Vec::new(),
            assignments: Vec::new(),
            inertia: f64::INFINITY,
            iteration_count: 0,
        }
    }

    /// Fit K-Means to data
    ///
    /// # Arguments
    /// * `data` - Matrix where each row is a data point
    ///
    /// # Returns
    /// Cluster assignments for each point
    ///
    /// # Examples
    /// ```
    /// use clustering::KMeans;
    /// use linear_algebra::matrix::Matrix;
    ///
    /// let data = Matrix::from_vec(vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0], 6, 1)?;
    /// let mut kmeans = KMeans::new(2, 100, 1e-4);
    /// let labels = kmeans.fit(&data);
    /// // labels = [0, 0, 0, 1, 1, 1] (two clusters)
    /// ```
    pub fn fit(&mut self, data: &Matrix<f64>) -> Vec<usize> {
        assert!(data.rows >= self.k,
            "Need at least K={} samples, got {}", self.k, data.rows);

        // 1. Initialize centroids
        self.centroids = self.initialize_centroids(data);
        self.iteration_count = 0;

        // 2. Iterate until convergence
        for iter in 0..self.max_iterations {
            // 2a. Assign points to nearest centroid
            let new_assignments = self.assign_points(data);

            // 2b. Update centroids
            let new_centroids = self.update_centroids(data, &new_assignments);

            // 2c. Check convergence (max centroid movement)
            let max_shift = self.compute_max_shift(&new_centroids);

            self.centroids = new_centroids;
            self.assignments = new_assignments;
            self.iteration_count = iter + 1;

            if max_shift < self.tolerance {
                break; // Converged
            }
        }

        // 3. Compute final inertia (quality metric)
        self.inertia = self.compute_inertia(data);

        self.assignments.clone()
    }

    /// Fit with animation (returns iterator over states)
    ///
    /// Useful for visualizing convergence step-by-step
    pub fn fit_animated<'a>(&'a mut self, data: &'a Matrix<f64>) -> KMeansIterator<'a> {
        self.centroids = self.initialize_centroids(data);
        self.iteration_count = 0;

        KMeansIterator {
            kmeans: self,
            data,
            iteration: 0,
        }
    }

    /// Predict cluster for new points
    pub fn predict(&self, data: &Matrix<f64>) -> Vec<usize> {
        (0..data.rows)
            .map(|i| {
                let point = data.row(i).unwrap();
                self.nearest_centroid(&point)
            })
            .collect()
    }

    // ===== Private methods =====

    fn initialize_centroids(&self, data: &Matrix<f64>) -> Vec<Vector<f64>> {
        match self.initialization {
            InitializationMethod::Random => self.random_init(data),
            InitializationMethod::KMeansPlusPlus => self.kmeans_plus_plus_init(data),
            InitializationMethod::Manual => panic!("Use set_initial_centroids() for manual init"),
        }
    }

    /// Random initialization (pick K random points)
    fn random_init(&self, data: &Matrix<f64>) -> Vec<Vector<f64>> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        let mut indices: Vec<usize> = (0..data.rows).collect();
        indices.shuffle(&mut rng);

        indices[..self.k]
            .iter()
            .map(|&i| data.row(i).unwrap())
            .collect()
    }

    /// K-Means++ initialization (smart seeding)
    ///
    /// Algorithm:
    /// 1. Pick first centroid uniformly at random
    /// 2. For each remaining centroid:
    ///    a. Compute distance D(x) from each point to nearest existing centroid
    ///    b. Pick next centroid with probability ∝ D(x)²
    ///
    /// Result: O(log k) approximation guarantee (vs random)
    fn kmeans_plus_plus_init(&self, data: &Matrix<f64>) -> Vec<Vector<f64>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut centroids = Vec::new();

        // 1. First centroid: random point
        let first_idx = rng.gen_range(0..data.rows);
        centroids.push(data.row(first_idx).unwrap());

        // 2. Remaining K-1 centroids
        for _ in 1..self.k {
            // Compute squared distance to nearest centroid for each point
            let mut distances_sq: Vec<f64> = (0..data.rows)
                .map(|i| {
                    let point = data.row(i).unwrap();
                    let min_dist = centroids.iter()
                        .map(|c| Self::euclidean_distance(&point, c))
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap();
                    min_dist * min_dist
                })
                .collect();

            // Weighted random selection (probability ∝ D(x)²)
            let total: f64 = distances_sq.iter().sum();
            let threshold = rng.gen::<f64>() * total;

            let mut cumsum = 0.0;
            let mut selected_idx = 0;
            for (i, &dist_sq) in distances_sq.iter().enumerate() {
                cumsum += dist_sq;
                if cumsum >= threshold {
                    selected_idx = i;
                    break;
                }
            }

            centroids.push(data.row(selected_idx).unwrap());
        }

        centroids
    }

    fn assign_points(&self, data: &Matrix<f64>) -> Vec<usize> {
        (0..data.rows)
            .map(|i| {
                let point = data.row(i).unwrap();
                self.nearest_centroid(&point)
            })
            .collect()
    }

    fn nearest_centroid(&self, point: &Vector<f64>) -> usize {
        self.centroids.iter()
            .enumerate()
            .map(|(idx, centroid)| (idx, Self::euclidean_distance(point, centroid)))
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
    }

    fn update_centroids(&self, data: &Matrix<f64>, assignments: &[usize]) -> Vec<Vector<f64>> {
        (0..self.k)
            .map(|cluster_id| {
                // Get all points in this cluster
                let cluster_points: Vec<Vector<f64>> = assignments.iter()
                    .enumerate()
                    .filter(|(_, &label)| label == cluster_id)
                    .map(|(i, _)| data.row(i).unwrap())
                    .collect();

                if cluster_points.is_empty() {
                    // Empty cluster: keep centroid as is
                    self.centroids[cluster_id].clone()
                } else {
                    // Compute mean of cluster points
                    Self::mean_vector(&cluster_points)
                }
            })
            .collect()
    }

    fn compute_max_shift(&self, new_centroids: &[Vector<f64>]) -> f64 {
        self.centroids.iter()
            .zip(new_centroids.iter())
            .map(|(old, new)| Self::euclidean_distance(old, new))
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    fn compute_inertia(&self, data: &Matrix<f64>) -> f64 {
        self.assignments.iter()
            .enumerate()
            .map(|(i, &cluster_id)| {
                let point = data.row(i).unwrap();
                let centroid = &self.centroids[cluster_id];
                let dist = Self::euclidean_distance(&point, centroid);
                dist * dist
            })
            .sum()
    }

    // Utility functions
    fn euclidean_distance(a: &Vector<f64>, b: &Vector<f64>) -> f64 {
        a.data.iter()
            .zip(b.data.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn mean_vector(vectors: &[Vector<f64>]) -> Vector<f64> {
        let dim = vectors[0].data.len();
        let n = vectors.len() as f64;

        let sum: Vec<f64> = (0..dim)
            .map(|i| {
                vectors.iter().map(|v| v.data[i]).sum::<f64>() / n
            })
            .collect();

        Vector { data: sum }
    }
}

/// Iterator for animated K-Means (returns state at each iteration)
pub struct KMeansIterator<'a> {
    kmeans: &'a mut KMeans,
    data: &'a Matrix<f64>,
    iteration: usize,
}

impl<'a> Iterator for KMeansIterator<'a> {
    type Item = KMeansState;

    fn next(&mut self) -> Option<Self::Item> {
        if self.iteration >= self.kmeans.max_iterations {
            return None;
        }

        // Perform one iteration
        let new_assignments = self.kmeans.assign_points(self.data);
        let new_centroids = self.kmeans.update_centroids(self.data, &new_assignments);
        let max_shift = self.kmeans.compute_max_shift(&new_centroids);

        self.kmeans.centroids = new_centroids.clone();
        self.kmeans.assignments = new_assignments.clone();
        self.kmeans.iteration_count = self.iteration + 1;
        self.iteration += 1;

        let state = KMeansState {
            centroids: new_centroids,
            assignments: new_assignments,
            iteration: self.iteration,
            max_shift,
            converged: max_shift < self.kmeans.tolerance,
        };

        if state.converged {
            None // Stop iteration
        } else {
            Some(state)
        }
    }
}

#[derive(Debug, Clone)]
pub struct KMeansState {
    pub centroids: Vec<Vector<f64>>,
    pub assignments: Vec<usize>,
    pub iteration: usize,
    pub max_shift: f64,
    pub converged: bool,
}
```

## Interactive Browser Demo

Create the most engaging ML visualization yet:

```rust
// web/src/components/kmeans_demo.rs
//! Interactive K-Means Clustering Demonstration
//!
//! **Features:**
//! - Click to add data points
//! - Adjust K (number of clusters)
//! - Watch centroids move iteration-by-iteration at 60 FPS
//! - Compare Random vs K-Means++ initialization
//! - Elbow method to choose optimal K

use dioxus::prelude::*;
use clustering::KMeans;
use linear_algebra::matrix::Matrix;

const CANVAS_WIDTH: f64 = 800.0;
const CANVAS_HEIGHT: f64 = 600.0;

#[component]
pub fn KMeansDemo() -> Element {
    // State
    let mut data_points = use_signal(|| Vec::<(f64, f64)>::new());
    let mut k = use_signal(|| 3);
    let mut is_running = use_signal(|| false);
    let mut current_state = use_signal(|| None::<KMeansState>);
    let mut kmeans = use_signal(|| KMeans::new(*k.read(), 100, 1e-3));

    // Canvas click handler (add point)
    let add_point = move |evt: MouseEvent| {
        let x = evt.client_x() as f64;
        let y = evt.client_y() as f64;
        data_points.write().push((x, y));
    };

    // Start clustering
    let start_clustering = move |_| {
        if data_points.read().len() < *k.read() {
            // Show error: need at least K points
            return;
        }

        // Convert points to Matrix
        let data = points_to_matrix(&data_points.read());

        // Create animated iterator
        let mut km = KMeans::new(*k.read(), 100, 1e-3);
        let states: Vec<_> = km.fit_animated(&data).collect();

        // Animate through states
        is_running.set(true);
        animate_states(states, current_state);
    };

    rsx! {
        div { class: "kmeans-demo",
            // Controls
            div { class: "controls",
                h2 { "K-Means Clustering" }

                // K slider
                label { "Number of Clusters (K): {k}" }
                input {
                    r#type: "range",
                    min: "1",
                    max: "10",
                    value: "{k}",
                    oninput: move |e| {
                        k.set(e.value.parse().unwrap());
                        kmeans.set(KMeans::new(*k.read(), 100, 1e-3));
                    }
                }

                // Buttons
                button {
                    onclick: start_clustering,
                    disabled: data_points.read().len() < *k.read(),
                    "Run K-Means"
                }

                button {
                    onclick: move |_| {
                        data_points.set(Vec::new());
                        current_state.set(None);
                    },
                    "Clear All"
                }

                button {
                    onclick: move |_| {
                        // Generate sample data (circles, blobs, etc.)
                        let points = generate_sample_data(100, *k.read());
                        data_points.set(points);
                    },
                    "Generate Sample Data"
                }

                p { "Click on canvas to add points" }
                p { "Points: {data_points.read().len()}" }

                if let Some(state) = current_state.read().as_ref() {
                    p { "Iteration: {state.iteration}" }
                    p { "Converged: {state.converged}" }
                }
            }

            // Canvas for visualization
            svg {
                width: "{CANVAS_WIDTH}",
                height: "{CANVAS_HEIGHT}",
                class: "canvas",
                onclick: add_point,

                // Data points (colored by cluster)
                for (i, (x, y)) in data_points.read().iter().enumerate() {
                    circle {
                        key: "{i}",
                        cx: "{x}",
                        cy: "{y}",
                        r: "6",
                        fill: "{get_cluster_color(i, &current_state.read())}",
                        stroke: "black",
                        stroke_width: "1",
                    }
                }

                // Centroids (large crosses)
                if let Some(state) = current_state.read().as_ref() {
                    for (cluster_id, centroid) in state.centroids.iter().enumerate() {
                        // Centroid marker (X shape)
                        g {
                            key: "centroid-{cluster_id}",
                            // Vertical line
                            line {
                                x1: "{centroid.data[0]}",
                                y1: "{centroid.data[1] - 15}",
                                x2: "{centroid.data[0]}",
                                y2: "{centroid.data[1] + 15}",
                                stroke: "{cluster_colors()[cluster_id]}",
                                stroke_width: "3",
                            }
                            // Horizontal line
                            line {
                                x1: "{centroid.data[0] - 15}",
                                y1: "{centroid.data[1]}",
                                x2: "{centroid.data[0] + 15}",
                                y2: "{centroid.data[1]}",
                                stroke: "{cluster_colors()[cluster_id]}",
                                stroke_width: "3",
                            }
                        }
                    }
                }

                // Cluster boundaries (Voronoi cells)
                if let Some(state) = current_state.read().as_ref() {
                    VoronoiDiagram { centroids: state.centroids.clone() }
                }
            }

            // Elbow method chart
            div { class: "elbow-chart",
                h3 { "Elbow Method (Choose K)" }
                ElbowPlot {
                    data: data_points.read().clone(),
                    max_k: 10,
                }
            }

            // Silhouette score
            if let Some(state) = current_state.read().as_ref() {
                div { class: "metrics",
                    h3 { "Cluster Quality" }
                    p { "Inertia: {compute_inertia(&data_points.read(), state):.2}" }
                    p { "Silhouette Score: {compute_silhouette(&data_points.read(), state):.2}" }
                }
            }
        }
    }
}

fn get_cluster_color(point_idx: usize, state: &Option<KMeansState>) -> String {
    match state {
        None => "gray".to_string(),
        Some(s) => {
            let cluster_id = s.assignments[point_idx];
            cluster_colors()[cluster_id].to_string()
        }
    }
}

fn cluster_colors() -> Vec<&'static str> {
    vec!["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
         "#1abc9c", "#34495e", "#e67e22", "#95a5a6", "#d35400"]
}

/// Animate through K-Means states at 60 FPS
fn animate_states(states: Vec<KMeansState>, current_state: Signal<Option<KMeansState>>) {
    let mut idx = 0;
    let interval = 1000 / 60; // 60 FPS

    use_effect(move || {
        if idx < states.len() {
            current_state.set(Some(states[idx].clone()));
            idx += 1;
            set_timeout(interval, /* recurse */);
        }
    });
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] K-Means algorithm implements Lloyd's algorithm correctly
- [ ] K-Means++ initialization produces better results than random (lower inertia)
- [ ] Converges within max_iterations or tolerance threshold
- [ ] Handles edge cases (K > n_samples, empty clusters)
- [ ] `fit()` returns cluster assignments
- [ ] `predict()` assigns new points to nearest centroid
- [ ] `fit_animated()` returns iterator for visualization

### Interactive Demo
- [ ] User can click to add points to canvas
- [ ] K slider adjusts number of clusters (1-10)
- [ ] "Run K-Means" button starts animation
- [ ] Centroids move smoothly iteration-by-iteration at 60 FPS
- [ ] Points change color as cluster assignments update
- [ ] Elbow method chart helps choose optimal K
- [ ] Silhouette score displayed for quality assessment
- [ ] Works on mobile (touch to add points)

### Quality Metrics
- [ ] Converges to same result as scikit-learn on test datasets
- [ ] K-Means++ reduces iterations to convergence by 30-50% vs random
- [ ] Browser demo runs at 60 FPS for <500 points
- [ ] Memory stable during long animations

## Success Metrics

**Technical:**
- Clustering quality matches scikit-learn (silhouette score within 1%)
- 100+ points clustered in <50ms
- Animation smooth at 60 FPS

**Educational:**
- 90%+ understand clustering concept after 5 min
- 80%+ can explain elbow method
- 70%+ understand K-Means++ advantage

---

**Estimated Effort:** 60-80 hours (2-3 weeks)
**Target Release:** v0.4.0
**Priority:** 🟢 MEDIUM-HIGH - Different paradigm, extremely visual
