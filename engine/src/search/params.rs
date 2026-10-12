//! Search constants. Most are runtime-tunable (exposed as UCI spin options
//! with the same names) so tests can try new values without rebuilding.

use std::sync::atomic::{AtomicI32, Ordering::Relaxed};

pub struct Tunable {
    pub name: &'static str,
    pub default: i32,
    pub min: i32,
    pub max: i32,
    value: AtomicI32,
}

impl Tunable {
    const fn new(name: &'static str, default: i32, min: i32, max: i32) -> Tunable {
        Tunable { name, default, min, max, value: AtomicI32::new(default) }
    }

    #[inline(always)]
    pub fn get(&self) -> i32 {
        self.value.load(Relaxed)
    }

    pub fn set(&self, value: i32) {
        self.value.store(value.clamp(self.min, self.max), Relaxed);
    }
}

macro_rules! tunables {
    ($($name:ident = $default:expr, $min:expr, $max:expr;)*) => {
        $(pub static $name: Tunable = Tunable::new(stringify!($name), $default, $min, $max);)*
        pub static TUNABLES: &[&Tunable] = &[$(&$name),*];
    };
}

tunables! {
    ASP_DELTA = 20, 5, 60;
    RFP_MAX_DEPTH = 8, 4, 12;
    RFP_MARGIN = 75, 30, 160;
    NMP_BASE_REDUCTION = 4, 2, 6;
    NMP_DEPTH_DIVISOR = 3, 2, 6;
    NMP_EVAL_DIVISOR = 150, 80, 400;
    LMP_BASE = 3, 1, 8;
    FP_MAX_DEPTH = 8, 4, 12;
    FP_BASE = 100, 30, 250;
    FP_PER_DEPTH = 90, 30, 200;
    HIST_PRUNE_PER_DEPTH = -2500, -6000, -500;
    SEE_QUIET_PER_DEPTH = -60, -150, -20;
    SEE_NOISY_PER_DEPTH = -100, -200, -30;
    SE_MIN_DEPTH = 8, 5, 12;
    SE_DOUBLE_MARGIN = 20, 5, 60;
    LMR_BASE_X100 = 75, 0, 150;
    LMR_DIVISOR_X100 = 230, 150, 400;
    LMR_HIST_DIVISOR = 8192, 2048, 32768;
    QS_FUTILITY_MARGIN = 150, 50, 400;
    HIST_BONUS_PER_DEPTH = 250, 100, 500;
    HIST_BONUS_OFFSET = 150, 0, 400;
    HIST_BONUS_MAX = 2000, 800, 4000;
    PROBCUT_MARGIN = 200, 80, 400;
}

pub fn find_tunable(name: &str) -> Option<&'static Tunable> {
    TUNABLES.iter().copied().find(|t| t.name.eq_ignore_ascii_case(name))
}

pub const ASP_MIN_DEPTH: i32 = 4;
pub const NMP_MIN_DEPTH: i32 = 3;
pub const IIR_MIN_DEPTH: i32 = 4;
pub const HIST_PRUNE_MAX_DEPTH: i32 = 4;
pub const SEE_PRUNE_MAX_DEPTH: i32 = 8;
pub const LMR_MIN_DEPTH: i32 = 3;
pub const PROBCUT_MIN_DEPTH: i32 = 5;
/// After a reduced search beats alpha: re-search a ply deeper if it won by
/// this much, a ply shallower if by less than the second margin.
pub const LMR_DEEPER_MARGIN: i32 = 60;
pub const LMR_SHALLOWER_MARGIN: i32 = 8;
pub const HIST_MAX: i32 = 16384;

/// Pawn-structure correction history.
pub const CORR_SIZE: usize = 16384;
pub const CORR_GRAIN: i32 = 256;
pub const CORR_MAX: i32 = CORR_GRAIN * 48;
