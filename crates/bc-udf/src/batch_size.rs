//! [`BatchSizeController`] — a PID governor for the dynamic batch size.
//!
//! Inference throughput is a trade-off: bigger batches amortize per-call overhead
//! and saturate the GPU, but raise per-batch latency. This controller watches the
//! measured per-batch latency and steers the [`Rebatcher`](crate::Rebatcher) target
//! toward a latency set-point — growing the batch when there is headroom, shrinking
//! it when batches run slow. It is the adaptive loop of the inference plane.
//!
//! The control law is a PID over the *relative* latency error
//! `(target − observed) / target`, applied multiplicatively to the current size, so
//! it is scale-free (works at 100 rows or 100k) and has a natural fixed point at
//! `observed == target`.

/// PID controller that maps observed per-batch latency to a target batch size.
pub struct BatchSizeController {
    target_latency_ms: f64,
    min_rows: usize,
    max_rows: usize,
    current: f64,
    kp: f64,
    ki: f64,
    kd: f64,
    integral: f64,
    prev_error: f64,
}

/// Cap per-step growth/shrink so a single anomalous latency can't swing the size wildly.
const MAX_STEP_FRACTION: f64 = 0.5;
/// Anti-windup clamp on the integral term.
const INTEGRAL_CLAMP: f64 = 5.0;

impl BatchSizeController {
    /// Create a controller targeting `target_latency_ms`, keeping the batch size in
    /// `[min_rows, max_rows]`, starting from `initial_rows`.
    pub fn new(
        target_latency_ms: f64,
        min_rows: usize,
        max_rows: usize,
        initial_rows: usize,
    ) -> Self {
        let min_rows = min_rows.max(1);
        let max_rows = max_rows.max(min_rows);
        let current = (initial_rows.max(min_rows).min(max_rows)) as f64;
        Self {
            target_latency_ms,
            min_rows,
            max_rows,
            current,
            kp: 0.4,
            ki: 0.05,
            kd: 0.1,
            integral: 0.0,
            prev_error: 0.0,
        }
    }

    /// The current target batch size in rows.
    pub fn current_rows(&self) -> usize {
        (self.current.round() as usize).clamp(self.min_rows, self.max_rows)
    }

    /// Feed the last batch's measured latency (ms) and return the new target size.
    ///
    /// `observed > target` (too slow) shrinks the batch; `observed < target`
    /// (headroom) grows it. Non-finite inputs and a non-positive target are ignored.
    pub fn update(&mut self, observed_latency_ms: f64) -> usize {
        if !observed_latency_ms.is_finite()
            || observed_latency_ms < 0.0
            || self.target_latency_ms <= 0.0
        {
            return self.current_rows();
        }

        let error = (self.target_latency_ms - observed_latency_ms) / self.target_latency_ms;
        self.integral = (self.integral + error).clamp(-INTEGRAL_CLAMP, INTEGRAL_CLAMP);
        let derivative = error - self.prev_error;
        self.prev_error = error;

        let adjustment = (self.kp * error + self.ki * self.integral + self.kd * derivative)
            .clamp(-MAX_STEP_FRACTION, MAX_STEP_FRACTION);

        self.current =
            (self.current * (1.0 + adjustment)).clamp(self.min_rows as f64, self.max_rows as f64);
        self.current_rows()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive the controller against a plant where latency ≈ `k · rows`, so the ideal
    /// size is `target / k`. Feeds back the size the controller just chose.
    fn converge(initial: usize, min: usize, max: usize, target_ms: f64, k: f64) -> usize {
        let mut ctrl = BatchSizeController::new(target_ms, min, max, initial);
        let mut rows = ctrl.current_rows();
        for _ in 0..200 {
            let observed = rows as f64 * k;
            assert!(
                (min..=max).contains(&ctrl.current_rows()),
                "size escaped bounds"
            );
            rows = ctrl.update(observed);
        }
        rows
    }

    #[test]
    fn grows_when_under_target() {
        // start small (fast batches) → controller should grow toward ideal 1000.
        let rows = converge(100, 50, 10_000, 10.0, 0.01);
        assert!(
            (800..=1200).contains(&rows),
            "did not converge near 1000: {rows}"
        );
    }

    #[test]
    fn shrinks_when_over_target() {
        // start large (slow batches) → controller should shrink toward ideal 1000.
        let rows = converge(10_000, 50, 10_000, 10.0, 0.01);
        assert!(
            (800..=1200).contains(&rows),
            "did not converge near 1000: {rows}"
        );
    }

    #[test]
    fn respects_bounds() {
        // Ideal size (2000) is above max → clamps at max and stays there.
        let mut ctrl = BatchSizeController::new(10.0, 100, 500, 100);
        for _ in 0..100 {
            let rows = ctrl.current_rows();
            ctrl.update(rows as f64 * 0.005); // latency model wants ~2000 rows
            assert!((100..=500).contains(&ctrl.current_rows()));
        }
        assert_eq!(ctrl.current_rows(), 500);
    }

    #[test]
    fn ignores_garbage_input() {
        let mut ctrl = BatchSizeController::new(10.0, 100, 1000, 400);
        let before = ctrl.current_rows();
        assert_eq!(ctrl.update(f64::NAN), before);
        assert_eq!(ctrl.update(-5.0), before);
        assert_eq!(ctrl.update(f64::INFINITY), before);
    }

    #[test]
    fn gains_match_python_pid_config() {
        // Parity guard: Python's `PIDConfig` defaults (kp=0.4, ki=0.05, kd=0.1,
        // integral_clamp=5.0, max_step_fraction=0.5) MUST match these gains — this
        // is the canonical batch-size controller both sides implement, and the
        // historical bug was the two diverging (ki/kd transposed in config).
        let ctrl = BatchSizeController::new(10.0, 1, 100, 10);
        assert_eq!((ctrl.kp, ctrl.ki, ctrl.kd), (0.4, 0.05, 0.1));
        assert_eq!(INTEGRAL_CLAMP, 5.0);
        assert_eq!(MAX_STEP_FRACTION, 0.5);
    }
}
