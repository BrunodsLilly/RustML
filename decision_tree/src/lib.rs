//! Decision Tree Classifier and Regressor
//!
//! Implementation of Classification and Regression Trees (CART) algorithm.
//! Supports both classification and regression tasks with various splitting criteria.

use linear_algebra::matrix::Matrix;
use ml_traits::supervised::SupervisedModel;
use ml_traits::Data;
use std::collections::HashMap;

mod tree;
pub use tree::*;

/// Decision Tree Classifier using CART algorithm
///
/// # Algorithm
/// 1. Start with entire dataset at root
/// 2. Find best split (feature + threshold) using Gini impurity or entropy
/// 3. Recursively split data until stopping criteria met
/// 4. Store majority class at leaf nodes
///
/// # Example
/// ```
/// use decision_tree::{DecisionTreeClassifier, SplitCriterion};
/// use linear_algebra::matrix::Matrix;
/// use ml_traits::supervised::SupervisedModel;
///
/// let X = Matrix::from_vec(
///     vec![
///         1.0, 2.0,  // Class 0
///         2.0, 3.0,  // Class 0
///         8.0, 9.0,  // Class 1
///         9.0, 10.0, // Class 1
///     ],
///     4, 2
/// ).unwrap();
/// let y = vec![0.0, 0.0, 1.0, 1.0];
///
/// let mut tree = DecisionTreeClassifier::new(
///     SplitCriterion::Gini,
///     Some(10),  // max depth
///     2,         // min samples split
///     1,         // min samples leaf
/// );
///
/// tree.fit(&X, &y).unwrap();
/// let predictions = tree.predict(&X).unwrap();
/// assert_eq!(predictions, vec![0, 0, 1, 1]);
/// ```
pub struct DecisionTreeClassifier {
    /// Splitting criterion (Gini or Entropy)
    criterion: SplitCriterion,
    /// Maximum tree depth (None = unlimited)
    max_depth: Option<usize>,
    /// Minimum samples required to split a node
    min_samples_split: usize,
    /// Minimum samples required at a leaf node
    min_samples_leaf: usize,
    /// Root node of the tree
    root: Option<DecisionNode>,
    /// Number of unique classes
    n_classes: usize,
    /// Whether the model has been fitted
    fitted: bool,
}

/// Splitting criterion for decision tree
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplitCriterion {
    /// Gini impurity: sum of p(1-p) for each class
    Gini,
    /// Entropy: -sum(p * log2(p)) for each class
    Entropy,
}

/// Decision tree node (recursive structure)
#[derive(Debug, Clone)]
pub struct DecisionNode {
    /// Feature index for split (None if leaf)
    feature_index: Option<usize>,
    /// Threshold value for split
    threshold: Option<f64>,
    /// Left child (samples <= threshold)
    left: Option<Box<DecisionNode>>,
    /// Right child (samples > threshold)
    right: Option<Box<DecisionNode>>,
    /// Class label (if leaf node)
    class_label: Option<usize>,
    /// Number of samples at this node
    n_samples: usize,
    /// Class distribution at this node
    class_counts: HashMap<usize, usize>,
}

impl DecisionTreeClassifier {
    /// Create a new Decision Tree Classifier
    ///
    /// # Arguments
    /// * `criterion` - Splitting criterion (Gini or Entropy)
    /// * `max_depth` - Maximum depth of tree (None = unlimited)
    /// * `min_samples_split` - Minimum samples to allow split
    /// * `min_samples_leaf` - Minimum samples required at leaf
    pub fn new(
        criterion: SplitCriterion,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
    ) -> Self {
        assert!(min_samples_split >= 2, "min_samples_split must be >= 2");
        assert!(min_samples_leaf >= 1, "min_samples_leaf must be >= 1");

        Self {
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            root: None,
            n_classes: 0,
            fitted: false,
        }
    }

    /// Build the decision tree recursively
    fn build_tree(
        &self,
        X: &Matrix<f64>,
        y: &[f64],
        indices: &[usize],
        depth: usize,
    ) -> DecisionNode {
        let n_samples = indices.len();

        // Calculate class counts
        let class_counts = Self::count_classes(y, indices);

        // Find majority class
        let majority_class = class_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&class, _)| class)
            .unwrap_or(0);

        // Stopping criteria
        let should_stop = n_samples < self.min_samples_split
            || self.max_depth.map_or(false, |max_d| depth >= max_d)
            || class_counts.len() == 1; // All samples same class

        if should_stop {
            return DecisionNode {
                feature_index: None,
                threshold: None,
                left: None,
                right: None,
                class_label: Some(majority_class),
                n_samples,
                class_counts,
            };
        }

        // Find best split
        if let Some((feature_idx, threshold, left_indices, right_indices)) =
            self.find_best_split(X, y, indices)
        {
            // Check min_samples_leaf constraint
            if left_indices.len() < self.min_samples_leaf
                || right_indices.len() < self.min_samples_leaf
            {
                // Create leaf node
                return DecisionNode {
                    feature_index: None,
                    threshold: None,
                    left: None,
                    right: None,
                    class_label: Some(majority_class),
                    n_samples,
                    class_counts,
                };
            }

            // Recursively build subtrees
            let left_child = self.build_tree(X, y, &left_indices, depth + 1);
            let right_child = self.build_tree(X, y, &right_indices, depth + 1);

            DecisionNode {
                feature_index: Some(feature_idx),
                threshold: Some(threshold),
                left: Some(Box::new(left_child)),
                right: Some(Box::new(right_child)),
                class_label: None,
                n_samples,
                class_counts,
            }
        } else {
            // No valid split found, create leaf
            DecisionNode {
                feature_index: None,
                threshold: None,
                left: None,
                right: None,
                class_label: Some(majority_class),
                n_samples,
                class_counts,
            }
        }
    }

    /// Find the best split for a node
    fn find_best_split(
        &self,
        X: &Matrix<f64>,
        y: &[f64],
        indices: &[usize],
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
        let n_features = X.n_features();
        let mut best_gain = 0.0;
        let mut best_split: Option<(usize, f64, Vec<usize>, Vec<usize>)> = None;

        // Calculate impurity of current node
        let parent_impurity = self.calculate_impurity(y, indices);

        // Try each feature
        for feature_idx in 0..n_features {
            // Get unique values for this feature
            let mut feature_values: Vec<f64> = indices
                .iter()
                .map(|&idx| *X.get(idx, feature_idx).expect("Valid index"))
                .collect();
            feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            feature_values.dedup();

            // Try each threshold (midpoint between consecutive values)
            for i in 0..feature_values.len().saturating_sub(1) {
                let threshold = (feature_values[i] + feature_values[i + 1]) / 2.0;

                // Split data
                let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
                    .iter()
                    .copied()
                    .partition(|&idx| *X.get(idx, feature_idx).expect("Valid index") <= threshold);

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }

                // Calculate information gain
                let left_impurity = self.calculate_impurity(y, &left_indices);
                let right_impurity = self.calculate_impurity(y, &right_indices);

                let n_left = left_indices.len() as f64;
                let n_right = right_indices.len() as f64;
                let n_total = indices.len() as f64;

                let weighted_impurity =
                    (n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity;

                let gain = parent_impurity - weighted_impurity;

                if gain > best_gain {
                    best_gain = gain;
                    best_split = Some((feature_idx, threshold, left_indices, right_indices));
                }
            }
        }

        best_split
    }

    /// Calculate impurity for a set of samples
    fn calculate_impurity(&self, y: &[f64], indices: &[usize]) -> f64 {
        let class_counts = Self::count_classes(y, indices);
        let n_samples = indices.len() as f64;

        match self.criterion {
            SplitCriterion::Gini => {
                // Gini impurity: 1 - sum(p_i^2)
                let mut gini = 1.0;
                for &count in class_counts.values() {
                    let p = count as f64 / n_samples;
                    gini -= p * p;
                }
                gini
            }
            SplitCriterion::Entropy => {
                // Entropy: -sum(p_i * log2(p_i))
                let mut entropy = 0.0;
                for &count in class_counts.values() {
                    if count > 0 {
                        let p = count as f64 / n_samples;
                        entropy -= p * p.log2();
                    }
                }
                entropy
            }
        }
    }

    /// Count class occurrences for given indices
    fn count_classes(y: &[f64], indices: &[usize]) -> HashMap<usize, usize> {
        let mut counts = HashMap::new();
        for &idx in indices {
            let class = y[idx] as usize;
            *counts.entry(class).or_insert(0) += 1;
        }
        counts
    }

    /// Predict class for a single sample
    fn predict_sample(&self, sample: &[f64], node: &DecisionNode) -> usize {
        // If leaf node, return class label
        if let Some(class) = node.class_label {
            return class;
        }

        // Otherwise, follow the tree
        let feature_idx = node.feature_index.expect("Internal node has feature");
        let threshold = node.threshold.expect("Internal node has threshold");

        if sample[feature_idx] <= threshold {
            self.predict_sample(sample, node.left.as_ref().expect("Left child exists"))
        } else {
            self.predict_sample(sample, node.right.as_ref().expect("Right child exists"))
        }
    }
}

impl SupervisedModel<f64, Matrix<f64>> for DecisionTreeClassifier {
    type Prediction = Vec<usize>;

    fn fit(&mut self, X: &Matrix<f64>, y: &[f64]) -> Result<(), String> {
        if X.n_samples() != y.len() {
            return Err(format!(
                "X has {} samples but y has {} labels",
                X.n_samples(),
                y.len()
            ));
        }

        if X.n_samples() < self.min_samples_split {
            return Err(format!(
                "Not enough samples ({}) to split (need {})",
                X.n_samples(),
                self.min_samples_split
            ));
        }

        // Determine number of classes
        let mut classes: Vec<usize> = y.iter().map(|&val| val as usize).collect();
        classes.sort_unstable();
        classes.dedup();
        self.n_classes = classes.len();

        if self.n_classes < 2 {
            return Err("Need at least 2 classes for classification".to_string());
        }

        // Build tree starting from root
        let all_indices: Vec<usize> = (0..X.n_samples()).collect();
        self.root = Some(self.build_tree(X, y, &all_indices, 0));
        self.fitted = true;

        Ok(())
    }

    fn predict(&self, X: &Matrix<f64>) -> Result<Self::Prediction, String> {
        if !self.fitted {
            return Err("Model not fitted. Call fit() first.".to_string());
        }

        let root = self.root.as_ref().expect("Root exists after fitting");
        let mut predictions = Vec::with_capacity(X.n_samples());

        for i in 0..X.n_samples() {
            let sample: Vec<f64> = (0..X.n_features())
                .map(|j| *X.get(i, j).expect("Valid index"))
                .collect();
            predictions.push(self.predict_sample(&sample, root));
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_decision_tree_basic() {
        // Simple linearly separable data
        let X = Matrix::from_vec(
            vec![
                1.0, 2.0, // Class 0
                2.0, 3.0, // Class 0
                8.0, 9.0, // Class 1
                9.0, 10.0, // Class 1
            ],
            4,
            2,
        )
        .unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];

        let mut tree = DecisionTreeClassifier::new(SplitCriterion::Gini, Some(10), 2, 1);
        tree.fit(&X, &y).unwrap();

        let predictions = tree.predict(&X).unwrap();
        assert_eq!(predictions, vec![0, 0, 1, 1]);
    }

    #[test]
    fn test_decision_tree_entropy() {
        // Test with entropy criterion
        let X = Matrix::from_vec(vec![1.0, 2.0, 2.0, 3.0, 8.0, 9.0, 9.0, 10.0], 4, 2).unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];

        let mut tree = DecisionTreeClassifier::new(SplitCriterion::Entropy, Some(10), 2, 1);
        tree.fit(&X, &y).unwrap();

        let predictions = tree.predict(&X).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_decision_tree_max_depth() {
        // Test max depth constraint
        let X = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
        let y = vec![0.0, 1.0, 2.0];

        let mut tree = DecisionTreeClassifier::new(SplitCriterion::Gini, Some(1), 2, 1);
        tree.fit(&X, &y).unwrap();

        let predictions = tree.predict(&X).unwrap();
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_decision_tree_not_fitted() {
        let tree = DecisionTreeClassifier::new(SplitCriterion::Gini, Some(10), 2, 1);
        let X = Matrix::from_vec(vec![1.0, 2.0], 1, 2).unwrap();

        let result = tree.predict(&X);
        assert!(result.is_err());
    }

    #[test]
    fn test_gini_impurity() {
        let tree = DecisionTreeClassifier::new(SplitCriterion::Gini, Some(10), 2, 1);

        // Pure node: all class 0
        let y = vec![0.0, 0.0, 0.0];
        let indices = vec![0, 1, 2];
        let impurity = tree.calculate_impurity(&y, &indices);
        assert_relative_eq!(impurity, 0.0, epsilon = 1e-10);

        // Perfectly impure: 50/50 split
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let indices = vec![0, 1, 2, 3];
        let impurity = tree.calculate_impurity(&y, &indices);
        assert_relative_eq!(impurity, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy_impurity() {
        let tree = DecisionTreeClassifier::new(SplitCriterion::Entropy, Some(10), 2, 1);

        // Pure node: all class 0
        let y = vec![0.0, 0.0, 0.0];
        let indices = vec![0, 1, 2];
        let impurity = tree.calculate_impurity(&y, &indices);
        assert_relative_eq!(impurity, 0.0, epsilon = 1e-10);

        // Perfectly impure: 50/50 split
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let indices = vec![0, 1, 2, 3];
        let impurity = tree.calculate_impurity(&y, &indices);
        assert_relative_eq!(impurity, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_min_samples_leaf() {
        // With min_samples_leaf=2, should not split if it would create leaf with 1 sample
        let X = Matrix::from_vec(vec![1.0, 2.0, 3.0, 10.0], 2, 2).unwrap();
        let y = vec![0.0, 1.0];

        let mut tree = DecisionTreeClassifier::new(SplitCriterion::Gini, Some(10), 2, 2);
        tree.fit(&X, &y).unwrap();

        // Tree should be just a leaf (can't split without violating min_samples_leaf)
        assert!(tree.root.as_ref().unwrap().class_label.is_some());
    }
}
