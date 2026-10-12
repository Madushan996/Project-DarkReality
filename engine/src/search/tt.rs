//! Lockless shared transposition table.
//!
//! Each entry is two atomics: `data` and `key ^ data`. A torn write from a
//! concurrent store makes the XOR check fail, so readers never trust a
//! half-written entry. Four 16-byte entries share one 64-byte cache line.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};

use crate::moves::Move;
use crate::search::{MATE_BOUND, MAX_PLY};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Bound {
    None = 0,
    Upper = 1,
    Lower = 2,
    Exact = 3,
}

#[derive(Clone, Copy, Debug)]
pub struct TTHit {
    pub mv: Move,
    pub score: i32,
    pub eval: i32,
    pub depth: i32,
    pub bound: Bound,
    /// Whether this position was on (or searched as) a principal variation.
    pub pv: bool,
}

#[derive(Default)]
struct Entry {
    key_xor_data: AtomicU64,
    data: AtomicU64,
}

const ENTRIES_PER_CLUSTER: usize = 4;

#[derive(Default)]
#[repr(align(64))]
struct Cluster {
    entries: [Entry; ENTRIES_PER_CLUSTER],
}

// data layout: move:16 | score:16 | eval:16 | depth:8 | bound:2 | age:5 | pv:1
const AGE_MASK: u8 = 31;

#[allow(clippy::too_many_arguments)]
fn pack(mv: Move, score: i32, eval: i32, depth: i32, bound: Bound, age: u8, pv: bool) -> u64 {
    mv.raw() as u64
        | (score as i16 as u16 as u64) << 16
        | (eval as i16 as u16 as u64) << 32
        | (depth.clamp(0, 255) as u64) << 48
        | (bound as u64) << 56
        | ((age & AGE_MASK) as u64) << 58
        | (pv as u64) << 63
}

fn depth_of(data: u64) -> i32 {
    ((data >> 48) & 0xFF) as i32
}

fn bound_of(data: u64) -> Bound {
    match (data >> 56) & 3 {
        1 => Bound::Upper,
        2 => Bound::Lower,
        3 => Bound::Exact,
        _ => Bound::None,
    }
}

fn age_of(data: u64) -> u8 {
    (data >> 58) as u8 & AGE_MASK
}

/// Mate scores are stored relative to the node, not the root.
pub fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score > MATE_BOUND {
        score + ply as i32
    } else if score < -MATE_BOUND {
        score - ply as i32
    } else {
        score
    }
}

pub fn score_from_tt(score: i32, ply: usize) -> i32 {
    if score > MATE_BOUND {
        score - ply as i32
    } else if score < -MATE_BOUND {
        score + ply as i32
    } else {
        score
    }
}

pub struct TT {
    clusters: Vec<Cluster>,
    age: AtomicU8,
}

impl TT {
    pub fn new(mb: usize) -> TT {
        let count = (mb.max(1) * 1024 * 1024 / std::mem::size_of::<Cluster>()).max(1);
        let mut clusters = Vec::with_capacity(count);
        clusters.resize_with(count, Cluster::default);
        TT { clusters, age: AtomicU8::new(0) }
    }

    pub fn clear(&self) {
        for cluster in &self.clusters {
            for e in &cluster.entries {
                e.key_xor_data.store(0, Relaxed);
                e.data.store(0, Relaxed);
            }
        }
        self.age.store(0, Relaxed);
    }

    /// Called once per `go` so stale entries are preferred for replacement.
    pub fn new_search(&self) {
        self.age.store((self.age.load(Relaxed) + 1) & AGE_MASK, Relaxed);
    }

    #[inline(always)]
    fn cluster(&self, hash: u64) -> &Cluster {
        let idx = ((hash as u128 * self.clusters.len() as u128) >> 64) as usize;
        &self.clusters[idx]
    }

    pub fn probe(&self, hash: u64, ply: usize) -> Option<TTHit> {
        for e in &self.cluster(hash).entries {
            let data = e.data.load(Relaxed);
            if e.key_xor_data.load(Relaxed) ^ data == hash && bound_of(data) != Bound::None {
                return Some(TTHit {
                    mv: Move::from_raw(data as u16),
                    score: score_from_tt((data >> 16) as u16 as i16 as i32, ply),
                    eval: (data >> 32) as u16 as i16 as i32,
                    depth: depth_of(data),
                    bound: bound_of(data),
                    pv: data >> 63 != 0,
                });
            }
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store(&self, hash: u64, mv: Move, score: i32, eval: i32, depth: i32, bound: Bound, ply: usize, pv: bool) {
        debug_assert!(ply <= MAX_PLY);
        let age = self.age.load(Relaxed);
        let cluster = self.cluster(hash);

        // Prefer the slot holding this position; otherwise evict the entry
        // that is shallowest after penalising stale generations.
        let mut slot = &cluster.entries[0];
        let mut slot_worth = i32::MAX;
        let mut existing = None;
        for e in &cluster.entries {
            let data = e.data.load(Relaxed);
            if e.key_xor_data.load(Relaxed) ^ data == hash {
                slot = e;
                existing = Some(data);
                break;
            }
            let stale = (age.wrapping_sub(age_of(data)) & AGE_MASK) as i32;
            let worth = depth_of(data) - 8 * stale;
            if worth < slot_worth {
                slot = e;
                slot_worth = worth;
            }
        }

        let mut mv = mv;
        if let Some(old) = existing {
            // Keep a deeper same-age result unless the new one is exact.
            if bound != Bound::Exact && depth + 4 < depth_of(old) && age_of(old) == age {
                return;
            }
            if mv.is_null() {
                mv = Move::from_raw(old as u16);
            }
        }

        let data = pack(mv, score_to_tt(score, ply), eval, depth, bound, age, pv);
        slot.data.store(data, Relaxed);
        slot.key_xor_data.store(hash ^ data, Relaxed);
    }

    /// Permille of sampled entries written during the current search.
    pub fn hashfull(&self) -> usize {
        let age = self.age.load(Relaxed);
        let sample = self.clusters.len().min(250);
        let mut used = 0;
        for cluster in &self.clusters[..sample] {
            for e in &cluster.entries {
                let data = e.data.load(Relaxed);
                if bound_of(data) != Bound::None && age_of(data) == age {
                    used += 1;
                }
            }
        }
        used * 1000 / (sample * ENTRIES_PER_CLUSTER)
    }
}
