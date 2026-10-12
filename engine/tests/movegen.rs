use darkreality::movegen::{generate_all, generate_noisy, generate_quiet, has_legal_move};
use darkreality::moves::MoveList;
use darkreality::perft::perft;
use darkreality::position::Position;

/// (FEN, [(depth, expected nodes)]) — the fast subset runs on every `cargo test`.
const SUITE: &[(&str, &[(u32, u64)])] = &[
    // Chess Programming Wiki standard positions.
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &[(1, 20), (2, 400), (3, 8902), (4, 197281), (5, 4865609)]),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", &[(1, 48), (2, 2039), (3, 97862), (4, 4085603)]),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", &[(1, 14), (2, 191), (3, 2812), (4, 43238), (5, 674624)]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", &[(1, 6), (2, 264), (3, 9467), (4, 422333)]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", &[(1, 44), (2, 1486), (3, 62379), (4, 2103487)]),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", &[(1, 46), (2, 2079), (3, 89890), (4, 3894594)]),
    // Edge cases: en passant pins, castling through check, promotions, stalemate.
    ("3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1", &[(6, 1134888)]),
    ("8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1", &[(6, 1015133)]),
    ("8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", &[(6, 1440467)]),
    ("5k2/8/8/8/8/8/8/4K2R w K - 0 1", &[(6, 661072)]),
    ("3k4/8/8/8/8/8/8/R3K3 w Q - 0 1", &[(6, 803711)]),
    ("r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1", &[(4, 1274206)]),
    ("r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1", &[(4, 1720476)]),
    ("2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1", &[(6, 3821001)]),
    ("8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1", &[(5, 1004658)]),
    ("4k3/1P6/8/8/8/8/K7/8 w - - 0 1", &[(6, 217342)]),
    ("8/P1k5/K7/8/8/8/8/8 w - - 0 1", &[(6, 92683)]),
    ("K1k5/8/P7/8/8/8/8/8 w - - 0 1", &[(6, 2217)]),
    ("8/k1P5/8/1K6/8/8/8/8 w - - 0 1", &[(7, 567584)]),
    ("8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1", &[(4, 23527)]),
    // Chess960 (Shredder-FEN castling fields).
    ("bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9", &[(1, 21), (2, 528), (3, 12189), (4, 326672)]),
    ("2nnrbkr/p1qppppp/8/1ppb4/6PP/3PP3/PPP2P2/BQNNRBKR w HEhe - 1 9", &[(1, 21), (2, 807), (3, 18002), (4, 667366)]),
    ("b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9", &[(1, 20), (2, 479), (3, 10471), (4, 273318)]),
];

/// Deeper counts; run with `cargo test --release -- --ignored`.
const DEEP: &[(&str, u32, u64)] = &[
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 5, 89941194),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 5, 164075551),
    ("bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9", 5, 8146062),
    ("2nnrbkr/p1qppppp/8/1ppb4/6PP/3PP3/PPP2P2/BQNNRBKR w HEhe - 1 9", 5, 16253601),
    ("b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9", 5, 6417013),
];

#[test]
fn perft_suite() {
    let mut failures = Vec::new();
    for (fen, cases) in SUITE {
        let pos = Position::from_fen(fen).unwrap();
        for &(depth, expected) in cases.iter() {
            let got = perft(&pos, depth);
            if got != expected {
                failures.push(format!("{fen} depth {depth}: expected {expected}, got {got}"));
            }
        }
    }
    assert!(failures.is_empty(), "perft mismatches:\n{}", failures.join("\n"));
}

#[test]
#[ignore]
fn perft_deep() {
    for &(fen, depth, expected) in DEEP {
        let pos = Position::from_fen(fen).unwrap();
        assert_eq!(perft(&pos, depth), expected, "{fen} depth {depth}");
    }
}

/// Walks the tree checking incremental state against from-scratch state.
fn check_tree(pos: &Position, depth: u32) {
    assert_eq!(pos.hash(), pos.compute_hash(), "hash drift at {}", pos.to_fen());

    let reparsed = Position::from_fen(&pos.to_fen()).unwrap();
    assert_eq!(reparsed.to_fen(), pos.to_fen());
    assert_eq!(reparsed.hash(), pos.hash(), "FEN round-trip hash differs for {}", pos.to_fen());
    assert_eq!(reparsed.checkers(), pos.checkers());
    assert_eq!(reparsed.pinned(), pos.pinned());

    let mut all = MoveList::new();
    let mut noisy = MoveList::new();
    let mut quiet = MoveList::new();
    generate_all(pos, &mut all);
    assert_eq!(has_legal_move(pos), !all.is_empty(), "terminal probe at {}", pos.to_fen());
    generate_noisy(pos, &mut noisy);
    generate_quiet(pos, &mut quiet);
    assert_eq!(noisy.len() + quiet.len(), all.len(), "noisy+quiet != all at {}", pos.to_fen());
    for mv in noisy.iter().chain(quiet.iter()) {
        assert!(all.contains(mv), "{mv:?} missing from full list at {}", pos.to_fen());
    }
    for mv in noisy.iter() {
        assert!(mv.is_capture() || mv.is_promotion(), "quiet move {mv:?} in noisy list");
    }

    if depth == 0 {
        return;
    }
    for &mv in all.iter() {
        check_tree(&pos.make_move(mv), depth - 1);
    }
}

#[test]
fn incremental_state_matches_scratch() {
    for (fen, _) in SUITE {
        check_tree(&Position::from_fen(fen).unwrap(), 3);
    }
}

#[test]
fn terminal_probe_handles_stalemate_check_and_pinned_pawns() {
    for fen in [
        "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1",
        "8/8/8/8/8/6q1/5k2/7K w - - 0 1",
        "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1",
        "4r2k/8/8/8/8/8/4P3/4K3 w - - 0 1",
        "4r2k/8/8/8/8/8/3P4/4K3 w - - 0 1",
        "K1k5/8/P7/8/8/8/8/8 w - - 0 1",
    ] {
        let pos = Position::from_fen(fen).unwrap();
        let all = darkreality::movegen::legal_moves(&pos);
        assert_eq!(has_legal_move(&pos), !all.is_empty(), "{fen}");
    }
}

#[test]
fn fen_round_trip_preserves_text() {
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w KQkq - 2 9",
        // Inner rook keeps its file letter; outermost rook is written as K/Q.
        "1r2k1r1/8/8/8/8/8/8/R1R1K3 w Ck - 0 1",
    ] {
        assert_eq!(Position::from_fen(fen).unwrap().to_fen(), fen);
    }
}

#[test]
fn rejects_bad_fens() {
    for fen in [
        "",
        "8/8/8/8/8/8/8/8 w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1",
        "rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w K - 0 1",
        "4k3/8/8/8/8/8/4R3/4K3 w - - 0 1",
    ] {
        assert!(Position::from_fen(fen).is_err(), "accepted bad FEN '{fen}'");
    }
}

/// `is_legal` must agree with the generator for every possible move code.
fn check_is_legal(pos: &Position, depth: u32, checked: &mut u64) {
    let all = darkreality::movegen::legal_moves(pos);
    for raw in 0..=u16::MAX {
        let mv = darkreality::moves::Move::from_raw(raw);
        let expected = all.contains(&mv);
        assert_eq!(darkreality::movegen::is_legal(pos, mv), expected, "{mv:?} in {}", pos.to_fen());
    }
    *checked += 1;
    if depth == 0 {
        return;
    }
    // Sample a few children to keep the test fast but varied.
    for &mv in all.iter().step_by(7) {
        check_is_legal(&pos.make_move(mv), depth - 1, checked);
    }
}

#[test]
fn is_legal_matches_generator() {
    let mut checked = 0;
    for (fen, _) in SUITE {
        check_is_legal(&Position::from_fen(fen).unwrap(), 2, &mut checked);
    }
    assert!(checked > 300, "only {checked} positions checked");
}
