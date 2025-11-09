//! Optimizer Race Controller - Manages concurrent optimizer execution
//!
//! This module coordinates multiple optimizers racing simultaneously on the same loss function.
//! It ensures synchronized timesteps, tracks performance metrics, and manages convergence.
//!
//! **Performance Critical:**
//! - Uses zero-allocation `step_2d()` pattern from optimizer.rs
//! - Bounded circular buffers for path history
//! - Real-time performance tracking (iterations/sec, loss reduction)

use super::loss_functions::LossFunction;
use neural_network::optimizer::Optimizer;

/// Maximum path points to store per optimizer (prevents memory leaks)
const MAX_PATH_LENGTH: usize = 1000;

/// Maximum loss history entries per optimizer
const MAX_LOSS_HISTORY: usize = 10000;

/// Convergence threshold - optimizer is "done" when loss < this value
const CONVERGENCE_THRESHOLD: f64 = 1e-6;

/// Maximum iterations before declaring non-convergence
const MAX_ITERATIONS: usize = 10000;

/// State for a single optimizer in the race
#[derive(Clone, Debug)]
pub struct RacerState {
    /// The optimizer instance (SGD, Momentum, RMSprop, or Adam)
    pub optimizer: Optimizer,

    /// Current position (x, y)
    pub position: (f64, f64),

    /// History of positions for 3D path rendering
    /// Stored as (x, y, z) where z = loss value
    pub path_3d: Vec<(f64, f64, f64)>,

    /// History of loss values
    pub losses: Vec<f64>,

    /// Current iteration number
    pub iteration: usize,

    /// Whether this optimizer has converged
    pub converged: bool,

    /// Display name (e.g., "SGD", "Adam")
    pub name: &'static str,

    /// Color for rendering (e.g., "#FF6B6B")
    pub color: &'static str,

    /// Rank in current race (1 = winning)
    pub rank: usize,
}

impl RacerState {
    /// Create a new racer with given optimizer and starting position
    pub fn new(
        optimizer: Optimizer,
        start: (f64, f64),
        name: &'static str,
        color: &'static str,
    ) -> Self {
        Self {
            optimizer,
            position: start,
            path_3d: Vec::new(),
            losses: Vec::new(),
            iteration: 0,
            converged: false,
            name,
            color,
            rank: 0,
        }
    }

    /// Reset to initial position
    pub fn reset(&mut self, start: (f64, f64)) {
        self.optimizer.reset();
        self.position = start;
        self.path_3d.clear();
        self.losses.clear();
        self.iteration = 0;
        self.converged = false;
        self.rank = 0;
    }

    /// Perform one optimization step
    pub fn step(&mut self, loss_fn: &LossFunction) {
        if self.converged || self.iteration >= MAX_ITERATIONS {
            return;
        }

        let (x, y) = self.position;
        let (dx, dy) = loss_fn.gradient(x, y);

        // Zero-allocation 2D optimization step
        self.position = self.optimizer.step_2d((x, y), (dx, dy));

        let (new_x, new_y) = self.position;
        let loss = loss_fn.evaluate(new_x, new_y);

        // Record 3D path (sample every 10 iterations to save memory)
        if self.iteration % 10 == 0 {
            if self.path_3d.len() >= MAX_PATH_LENGTH {
                self.path_3d.remove(0); // Remove oldest point
            }
            self.path_3d.push((new_x, new_y, loss));
        }

        // Record loss history
        if self.losses.len() >= MAX_LOSS_HISTORY {
            self.losses.remove(0); // Remove oldest loss
        }
        self.losses.push(loss);

        self.iteration += 1;

        // Check convergence
        if loss < CONVERGENCE_THRESHOLD {
            self.converged = true;
        }
    }

    /// Get current loss value
    pub fn current_loss(&self) -> f64 {
        self.losses.last().copied().unwrap_or(f64::INFINITY)
    }

    /// Get loss reduction percentage from start
    pub fn loss_reduction_pct(&self) -> f64 {
        if self.losses.len() < 2 {
            return 0.0;
        }
        let initial = self.losses[0];
        let current = self.current_loss();
        if initial == 0.0 {
            return 100.0;
        }
        ((initial - current) / initial * 100.0).max(0.0).min(100.0)
    }

    /// Get average iterations per second (rough estimate)
    pub fn iterations_per_sec(&self) -> f64 {
        // This is a placeholder - in real implementation would track wall time
        self.iteration as f64
    }
}

/// Controller for the full race with 4 optimizers
#[derive(Clone, Debug)]
pub struct RaceController {
    /// The 4 competing optimizers
    pub racers: Vec<RacerState>,

    /// Current loss function being optimized
    pub loss_fn: LossFunction,

    /// Total race iterations completed
    pub total_iterations: usize,

    /// Whether race is currently running
    pub is_running: bool,
}

impl RaceController {
    /// Create a new race with 4 standard optimizers (SGD, Momentum, RMSprop, Adam)
    pub fn new(loss_fn: LossFunction, learning_rate: f64, start: (f64, f64)) -> Self {
        let racers = vec![
            RacerState::new(
                Optimizer::sgd(learning_rate),
                start,
                "SGD",
                "#FF6B6B", // Red
            ),
            RacerState::new(
                Optimizer::momentum(learning_rate, 0.9),
                start,
                "Momentum",
                "#FFA500", // Orange
            ),
            RacerState::new(
                Optimizer::rmsprop(learning_rate, 0.9, 1e-8),
                start,
                "RMSprop",
                "#4ECDC4", // Teal/Green
            ),
            RacerState::new(
                Optimizer::adam(learning_rate, 0.9, 0.999, 1e-8),
                start,
                "Adam",
                "#667EEA", // Blue
            ),
        ];

        Self {
            racers,
            loss_fn,
            total_iterations: 0,
            is_running: false,
        }
    }

    /// Reset all racers to starting position
    pub fn reset(&mut self, start: (f64, f64)) {
        for racer in &mut self.racers {
            racer.reset(start);
        }
        self.total_iterations = 0;
    }

    /// Perform one synchronized step for all racers
    pub fn step_all(&mut self) {
        for racer in &mut self.racers {
            racer.step(&self.loss_fn);
        }
        self.total_iterations += 1;
        self.update_rankings();
    }

    /// Perform N synchronized steps
    pub fn step_n(&mut self, n: usize) {
        for _ in 0..n {
            self.step_all();
        }
    }

    /// Update rankings based on current loss values
    fn update_rankings(&mut self) {
        // Sort by current loss (lowest = best = rank 1)
        let mut indexed_losses: Vec<(usize, f64)> = self
            .racers
            .iter()
            .enumerate()
            .map(|(i, r)| (i, r.current_loss()))
            .collect();

        indexed_losses.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assign ranks
        for (rank, (racer_idx, _)) in indexed_losses.iter().enumerate() {
            self.racers[*racer_idx].rank = rank + 1;
        }
    }

    /// Check if any racer has won (converged first)
    pub fn has_winner(&self) -> bool {
        self.racers.iter().any(|r| r.converged)
    }

    /// Get the winning racer (first to converge, or current leader)
    pub fn get_winner(&self) -> Option<&RacerState> {
        // First check for convergence
        if let Some(winner) = self.racers.iter().find(|r| r.converged) {
            return Some(winner);
        }

        // Otherwise return current leader (lowest loss)
        self.racers.iter().min_by(|a, b| {
            a.current_loss()
                .partial_cmp(&b.current_loss())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get sorted racers by rank
    pub fn get_leaderboard(&self) -> Vec<&RacerState> {
        let mut sorted = self.racers.iter().collect::<Vec<_>>();
        sorted.sort_by_key(|r| r.rank);
        sorted
    }

    /// Check if all racers have finished (converged or hit max iterations)
    pub fn is_finished(&self) -> bool {
        self.racers
            .iter()
            .all(|r| r.converged || r.iteration >= MAX_ITERATIONS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_race_controller_creation() {
        let controller = RaceController::new(LossFunction::Quadratic, 0.01, (1.0, 1.0));
        assert_eq!(controller.racers.len(), 4);
        assert_eq!(controller.racers[0].name, "SGD");
        assert_eq!(controller.racers[3].name, "Adam");
    }

    #[test]
    fn test_race_step() {
        let mut controller = RaceController::new(LossFunction::Quadratic, 0.1, (1.0, 1.0));
        controller.step_all();

        // All racers should have moved
        for racer in &controller.racers {
            assert_eq!(racer.iteration, 1);
            assert_eq!(racer.losses.len(), 1);
        }
    }

    #[test]
    fn test_rankings() {
        let mut controller = RaceController::new(LossFunction::Quadratic, 0.1, (1.0, 1.0));
        controller.step_n(100);

        // Adam should typically win on simple quadratic
        let leaderboard = controller.get_leaderboard();
        assert!(leaderboard[0].rank == 1);

        // All ranks should be assigned
        let ranks: Vec<usize> = leaderboard.iter().map(|r| r.rank).collect();
        assert_eq!(ranks, vec![1, 2, 3, 4]);
    }
}
