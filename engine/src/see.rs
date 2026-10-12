//! Static exchange evaluation using the swap-list algorithm.

use crate::attacks::{bishop_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::eval::PIECE_VALUES;
use crate::moves::Move;
use crate::position::Position;
use crate::types::{Color, PieceType, Square};

/// Value used for the king inside exchanges: capturing it ends the sequence.
const KING_VALUE: i32 = 20_000;

#[inline(always)]
fn value(pt: PieceType) -> i32 {
    if pt == PieceType::King { KING_VALUE } else { PIECE_VALUES[pt.idx()] }
}

fn least_valuable(pos: &Position, attackers: Bitboard) -> Option<(PieceType, Square)> {
    PieceType::ALL.iter().find_map(|&pt| {
        let set = attackers & pos.pieces(pt);
        set.any().then(|| (pt, set.lsb()))
    })
}

/// Material balance of the capture sequence on `mv.to()` for the mover,
/// assuming both sides recapture with their least valuable piece and may
/// stop at any point. Pins are ignored.
pub fn see(pos: &Position, mv: Move) -> i32 {
    if mv.is_castle() {
        return 0;
    }
    let from = mv.from();
    let to = mv.to();
    let us: Color = pos.side_to_move();

    let mut gain = [0i32; 34];
    let mut occ = pos.occupied() ^ from.bb();
    let mut on_square = value(pos.piece_on(from).piece_type());

    if mv.is_en_passant() {
        gain[0] = PIECE_VALUES[PieceType::Pawn.idx()];
        occ ^= Square::new(to.raw() ^ 8).bb();
    } else if mv.is_capture() {
        gain[0] = value(pos.piece_on(to).piece_type());
    }
    if let Some(promo) = mv.promo_piece() {
        gain[0] += value(promo) - PIECE_VALUES[PieceType::Pawn.idx()];
        on_square = value(promo);
    }

    let diagonal = pos.pieces(PieceType::Bishop) | pos.pieces(PieceType::Queen);
    let orthogonal = pos.pieces(PieceType::Rook) | pos.pieces(PieceType::Queen);
    let mut attackers = pos.attackers_to(to, occ) & occ;
    let mut side = !us;
    let mut depth = 0;

    while let Some((pt, sq)) = least_valuable(pos, attackers & pos.colored(side)) {
        // A king may not capture onto a square the opponent still defends.
        if pt == PieceType::King && (attackers & pos.colored(!side)).any() {
            break;
        }
        depth += 1;
        gain[depth] = on_square - gain[depth - 1];
        // Neither standing pat nor capturing helps this side: the earlier
        // result cannot change, so stop and drop the speculative entry.
        if (-gain[depth - 1]).max(gain[depth]) < 0 {
            depth -= 1;
            break;
        }
        occ ^= sq.bb();
        if matches!(pt, PieceType::Pawn | PieceType::Bishop | PieceType::Queen) {
            attackers |= bishop_attacks(to, occ) & diagonal;
        }
        if matches!(pt, PieceType::Rook | PieceType::Queen) {
            attackers |= rook_attacks(to, occ) & orthogonal;
        }
        attackers &= occ;
        on_square = value(pt);
        side = !side;
    }

    while depth > 0 {
        gain[depth - 1] = -(-gain[depth - 1]).max(gain[depth]);
        depth -= 1;
    }
    gain[0]
}

#[inline]
pub fn see_ge(pos: &Position, mv: Move, threshold: i32) -> bool {
    if mv.is_castle() {
        return threshold <= 0;
    }
    let mut immediate = if mv.is_en_passant() {
        PIECE_VALUES[PieceType::Pawn.idx()]
    } else if mv.is_capture() {
        value(pos.piece_on(mv.to()).piece_type())
    } else {
        0
    };
    let occupant = if let Some(promo) = mv.promo_piece() {
        immediate += value(promo) - PIECE_VALUES[PieceType::Pawn.idx()];
        value(promo)
    } else {
        value(pos.piece_on(mv.from()).piece_type())
    };
    // The opponent can decline to recapture, so the immediate gain is an
    // upper bound. Losing our occupant without recovering anything is the
    // lower bound; the swap-list's later exchanges can only improve on it.
    if immediate < threshold {
        return false;
    }
    if immediate - occupant >= threshold {
        return true;
    }
    see(pos, mv) >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::legal_moves;

    fn compare(pos: &Position, depth: u32, checked: &mut u64) {
        for &mv in legal_moves(pos).iter() {
            let score = see(pos, mv);
            for threshold in [i32::MIN, -20000, -1500, -900, -500, -300,
                              -120, -60, -1, 0, 1, 100, 300, 500, 900,
                              1500, 20000, i32::MAX, score - 1, score, score + 1] {
                assert_eq!(see_ge(pos, mv, threshold), score >= threshold,
                           "{} {} threshold {} full SEE {}", pos.to_fen(), mv.to_uci(false), threshold, score);
                *checked += 1;
            }
            if depth > 0 {
                compare(&pos.make_move(mv), depth - 1, checked);
            }
        }
    }

    #[test]
    fn material_bounds_preserve_full_exchange_decisions() {
        let mut checked = 0;
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
            "k6r/6P1/8/8/8/8/8/K7 w - - 0 1",
            "k7/8/8/8/8/8/6p1/K6R b - - 0 1",
            "3k4/8/8/8/3n4/2P5/8/3RK3 w - - 0 1",
            "4k3/8/3p4/4p3/3R4/8/8/4K3 w - - 0 1",
        ] {
            compare(&Position::from_fen(fen).unwrap(), 2, &mut checked);
        }
        assert!(checked > 100_000, "too little exchange coverage: {checked}");
        println!("Compared {checked} exchange-threshold decisions with the unchanged full swap-list");
    }
}
