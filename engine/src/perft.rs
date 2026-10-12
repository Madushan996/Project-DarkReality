use std::time::Instant;

use crate::movegen::generate_all;
use crate::moves::MoveList;
use crate::position::Position;

/// Counts leaf nodes at `depth`, using bulk counting at depth 1.
pub fn perft(pos: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut list = MoveList::new();
    generate_all(pos, &mut list);
    if depth == 1 {
        return list.len() as u64;
    }
    list.iter().map(|&mv| perft(&pos.make_move(mv), depth - 1)).sum()
}

/// Perft with a per-root-move breakdown, for debugging against other tools.
pub fn divide(pos: &Position, depth: u32, chess960: bool) -> u64 {
    let start = Instant::now();
    let mut total = 0;
    if depth > 0 {
        let mut list = MoveList::new();
        generate_all(pos, &mut list);
        for &mv in list.iter() {
            let nodes = perft(&pos.make_move(mv), depth - 1);
            println!("{}: {nodes}", mv.to_uci(chess960));
            total += nodes;
        }
    } else {
        total = 1;
    }
    let secs = start.elapsed().as_secs_f64();
    println!();
    println!("Nodes searched: {total}");
    println!("Time: {:.3}s  ({:.1} Mnps)", secs, total as f64 / secs.max(1e-9) / 1e6);
    total
}
