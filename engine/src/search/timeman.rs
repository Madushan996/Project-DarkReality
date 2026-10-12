use std::time::Instant;

use crate::search::Limits;
use crate::types::Color;

/// Soft limit: don't start a new iteration past it.
/// Hard limit: abort the search mid-iteration.
pub struct TimeManager {
    start: Instant,
    soft_ms: Option<u64>,
    hard_ms: Option<u64>,
}

impl TimeManager {
    pub fn new(limits: &Limits, stm: Color, overhead_ms: u64) -> TimeManager {
        let start = Instant::now();
        if let Some(movetime) = limits.movetime {
            let t = movetime.saturating_sub(overhead_ms).max(1);
            return TimeManager { start, soft_ms: Some(t), hard_ms: Some(t) };
        }
        let Some(time) = limits.time[stm.idx()] else {
            return TimeManager { start, soft_ms: None, hard_ms: None };
        };
        let inc = limits.inc[stm.idx()];
        let available = time.saturating_sub(overhead_ms).max(1);
        let base = match limits.movestogo {
            Some(mtg) => available / mtg.clamp(1, 50),
            None => available / 20 + inc * 3 / 4,
        };
        let hard = (base * 3).min(available * 3 / 4).max(1);
        let soft = (base * 7 / 10).min(hard);
        TimeManager { start, soft_ms: Some(soft), hard_ms: Some(hard) }
    }

    #[inline]
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    #[inline]
    pub fn soft_exceeded(&self) -> bool {
        self.soft_exceeded_scaled(1.0)
    }

    /// Soft limit scaled by how settled the search looks (see `stop_scale`).
    #[inline]
    pub fn soft_exceeded_scaled(&self, scale: f64) -> bool {
        self.soft_ms.is_some_and(|t| self.elapsed_ms() as f64 >= t as f64 * scale)
    }

    #[inline]
    pub fn hard_exceeded(&self) -> bool {
        self.hard_ms.is_some_and(|t| self.elapsed_ms() >= t)
    }
}

/// Multiplier for the soft limit after an iteration.
///
/// `stability`: consecutive iterations with the same best move.
/// `best_fraction`: share of this iteration's root nodes spent on that move.
/// A long-stable move that soaked up most of the effort is very likely
/// right, so stop early; a fresh or contested choice gets more time.
pub fn stop_scale(stability: u32, best_fraction: f64) -> f64 {
    const STABILITY: [f64; 5] = [2.2, 1.4, 1.1, 0.9, 0.8];
    let stability = STABILITY[(stability as usize).min(STABILITY.len() - 1)];
    let effort = (1.6 - best_fraction).clamp(0.5, 1.6);
    stability * effort
}
