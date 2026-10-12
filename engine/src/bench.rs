//! `bench`: fixed-depth searches over a fixed suite.
//!
//! The total node count is a fingerprint of the search: any functional change
//! alters it, and OpenBench uses it to verify builds. Record it in every
//! commit message that changes search behaviour.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use crate::position::Position;
use crate::search::tt::TT;
use crate::search::{Limits, Worker};

pub const DEFAULT_DEPTH: i32 = 11;

pub const BENCH_FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 5",
    "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 4 8",
    "r1b1k2r/ppppnppp/2n2q2/2b5/3NP3/2P1B3/PP3PPP/RN1QKB1R w KQkq - 3 7",
    "2r3k1/1q3ppp/p3p3/1p1nP3/3P4/P2Q1N2/1P3PPP/2R3K1 b - - 0 25",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "8/5pk1/6p1/8/8/6P1/5PK1/3R4 w - - 0 1",
    "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",
    "8/8/3k4/8/8/3K4/3Q4/8 w - - 0 1",
    "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
];

pub fn run(depth: i32) {
    let tt = Arc::new(TT::new(16));
    let mut worker = Worker::new(tt.clone(), Arc::new(AtomicBool::new(false)));
    worker.silent = true;

    let start = Instant::now();
    let mut nodes = 0;
    for fen in BENCH_FENS {
        let pos = Position::from_fen(fen).expect("bench FEN is valid");
        tt.clear();
        worker.clear();
        let limits = Limits { depth: Some(depth), ..Limits::default() };
        nodes += worker.search(&pos, &[pos.hash()], limits).nodes;
    }
    let secs = start.elapsed().as_secs_f64();
    println!("{nodes} nodes {} nps", (nodes as f64 / secs.max(1e-9)) as u64);
}
