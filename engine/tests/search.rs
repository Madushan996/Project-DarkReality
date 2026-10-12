use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use darkreality::movegen::legal_moves;
use darkreality::position::Position;
use darkreality::search::tt::TT;
use darkreality::search::{Limits, MATE, MATE_BOUND, SearchResult, Worker};
use darkreality::see::see;

fn search(fen: &str, depth: i32) -> SearchResult {
    let pos = Position::from_fen(fen).unwrap();
    let mut worker = Worker::new(Arc::new(TT::new(8)), Arc::new(AtomicBool::new(false)));
    worker.silent = true;
    worker.search(&pos, &[pos.hash()], Limits { depth: Some(depth), ..Limits::default() })
}

fn find_move(fen: &str, uci: &str) -> darkreality::moves::Move {
    let pos = Position::from_fen(fen).unwrap();
    legal_moves(&pos).iter().copied().find(|m| m.to_uci(false) == uci).expect("legal move")
}

#[test]
fn finds_mate_in_one() {
    let r = search("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", 6);
    assert_eq!(r.best_move.to_uci(false), "d1d8");
    assert_eq!(r.score, MATE - 1);
}

#[test]
fn mates_instead_of_stalemating() {
    // Qf7 stalemates; Qd8 / Qh7 mate.
    let r = search("7k/3Q4/6K1/8/8/8/8/8 w - - 0 1", 6);
    assert_eq!(r.score, MATE - 1, "best {}", r.best_move);
}

#[test]
fn converts_kqk() {
    let r = search("8/8/3k4/8/8/3K4/3Q4/8 w - - 0 1", 24);
    assert!(r.score > MATE_BOUND, "no mate found: {}", r.score);
}

#[test]
fn defends_against_mate_threat() {
    // Black must stop Qxf7#; grabbing the queen with Nxh5 does it.
    let r = search("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 4 4", 8);
    assert_eq!(r.best_move.to_uci(false), "f6h5");
}

#[test]
fn stalemate_root_has_no_move() {
    let r = search("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 5);
    assert!(r.best_move.is_null());
}

#[test]
fn insufficient_material_scores_draw() {
    let r = search("8/8/4k3/8/8/3NK3/8/8 w - - 0 1", 10);
    assert_eq!(r.score, 0);
}

#[test]
fn quiet_horizon_observes_fifty_move_draw() {
    // Every move advances the clock to 100; none can mate or reset it.
    let r = search("7k/8/8/8/8/2K5/Q7/8 w - - 99 1", 1);
    assert_eq!(r.score, 0, "a quiet horizon must not hide the draw");
}

#[test]
fn mate_takes_precedence_at_fifty_move_boundary() {
    let r = search("7k/3Q4/6K1/8/8/8/8/8 w - - 99 1", 6);
    assert_eq!(r.score, MATE - 1, "the mating move must not be scored as a draw");
}

#[test]
fn see_values() {
    let fen = "1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1";
    assert_eq!(see(&Position::from_fen(fen).unwrap(), find_move(fen, "e1e5")), 100);

    let fen = "1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1";
    assert_eq!(see(&Position::from_fen(fen).unwrap(), find_move(fen, "d3e5")), -200);

    // Quiet moves: safe square is 0, pawn-attacked square loses the rook.
    let fen = "4k3/8/8/3p4/8/8/8/3RK3 w - - 0 1";
    assert_eq!(see(&Position::from_fen(fen).unwrap(), find_move(fen, "d1d4")), 0);
    let fen = "4k3/8/2p5/8/8/8/8/3RK3 w - - 0 1";
    assert_eq!(see(&Position::from_fen(fen).unwrap(), find_move(fen, "d1d5")), -500);

    // Undefended pawn; then a king may not recapture onto a defended square.
    let fen = "4k3/8/3p4/8/8/8/8/3RK3 w - - 0 1";
    assert_eq!(see(&Position::from_fen(fen).unwrap(), find_move(fen, "d1d6")), 100);
    let fen = "8/8/3pk3/8/8/8/8/3RK3 w - - 0 1";
    assert_eq!(see(&Position::from_fen(fen).unwrap(), find_move(fen, "d1d6")), -400);
    let fen = "8/8/3pk3/8/8/8/3R4/3RK3 w - - 0 1";
    assert_eq!(see(&Position::from_fen(fen).unwrap(), find_move(fen, "d2d6")), 100);
}

#[test]
fn bench_is_deterministic() {
    let a = search("r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 4 8", 10);
    let b = search("r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 4 8", 10);
    assert_eq!((a.nodes, a.best_move, a.score), (b.nodes, b.best_move, b.score));
}
