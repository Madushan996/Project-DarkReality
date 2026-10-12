//! Fully legal move generation using check and pin masks.
//!
//! Moves are split into *noisy* (captures, en passant, queen promotions) and
//! *quiet* (everything else, including under-promotions and castling) so the
//! search can generate them in stages. Noisy ∪ quiet = all legal moves.

use crate::attacks::*;
use crate::bitboard::Bitboard;
use crate::moves::{Move, MoveList};
use crate::position::{Position, castle_index};
use crate::types::{Color, PieceType, Square};

pub fn generate_all(pos: &Position, list: &mut MoveList) {
    generate::<true, true, false>(pos, list);
}

pub fn generate_noisy(pos: &Position, list: &mut MoveList) {
    generate::<true, false, false>(pos, list);
}

pub fn generate_quiet(pos: &Position, list: &mut MoveList) {
    generate::<false, true, false>(pos, list);
}

pub fn legal_moves(pos: &Position) -> MoveList {
    let mut list = MoveList::new();
    generate_all(pos, &mut list);
    list
}

/// Terminal-position probe. Outside check, an unpinned pawn's unobstructed
/// advance already proves that the side can move. Otherwise stop generation
/// as soon as any legal move is found, using the same check/pin masks.
#[inline]
pub fn has_legal_move(pos: &Position) -> bool {
    if !pos.in_check() {
        let pawns = pos.pieces_of(pos.side_to_move(), PieceType::Pawn) & !pos.pinned();
        if (pawns.forward(pos.side_to_move()) & !pos.occupied()).any() {
            return true;
        }
    }
    let mut list = MoveList::new();
    generate::<true, true, true>(pos, &mut list);
    !list.is_empty()
}

#[inline(always)]
fn push_moves(list: &mut MoveList, from: Square, targets: Bitboard, theirs: Bitboard) {
    for to in targets {
        let flag = if theirs.contains(to) { Move::CAPTURE } else { Move::QUIET };
        list.push(Move::new(from, to, flag));
    }
}

fn generate<const NOISY: bool, const QUIET: bool, const FIRST_ONLY: bool>(pos: &Position, list: &mut MoveList) {
    use PieceType::*;
    let us = pos.side_to_move();
    let them = !us;
    let occ = pos.occupied();
    let theirs = pos.colored(them);
    let ksq = pos.king_sq(us);
    let checkers = pos.checkers();
    let pinned = pos.pinned();

    let mut targets = Bitboard::EMPTY;
    if NOISY {
        targets |= theirs;
    }
    if QUIET {
        targets |= !occ;
    }

    // King: test destination safety with the king lifted off the board so
    // sliders "see through" its current square.
    let occ_without_king = occ ^ ksq.bb();
    for to in king_attacks(ksq) & targets {
        if !pos.is_attacked_by(to, them, occ_without_king) {
            let flag = if theirs.contains(to) { Move::CAPTURE } else { Move::QUIET };
            list.push(Move::new(ksq, to, flag));
            if FIRST_ONLY { return; }
        }
    }

    if checkers.more_than_one() {
        return;
    }
    // In check: other pieces must capture the checker or block.
    let check_mask = if checkers.any() { between(ksq, checkers.lsb()) | checkers } else { Bitboard::ALL };
    let targets = targets & check_mask;

    // Pinned knights can never move.
    for from in pos.pieces_of(us, Knight) & !pinned {
        push_moves(list, from, knight_attacks(from) & targets, theirs);
        if FIRST_ONLY && !list.is_empty() { return; }
    }

    for from in pos.pieces_of(us, Bishop) | pos.pieces_of(us, Queen) {
        let mut t = bishop_attacks(from, occ) & targets;
        if pinned.contains(from) {
            t &= line(ksq, from);
        }
        push_moves(list, from, t, theirs);
        if FIRST_ONLY && !list.is_empty() { return; }
    }

    for from in pos.pieces_of(us, Rook) | pos.pieces_of(us, Queen) {
        let mut t = rook_attacks(from, occ) & targets;
        if pinned.contains(from) {
            t &= line(ksq, from);
        }
        push_moves(list, from, t, theirs);
        if FIRST_ONLY && !list.is_empty() { return; }
    }

    generate_pawns::<NOISY, QUIET>(pos, list, us, ksq, check_mask);
    if FIRST_ONLY && !list.is_empty() { return; }

    if QUIET && checkers.is_empty() {
        generate_castling(pos, list, us, ksq);
    }
}

#[inline(always)]
fn generate_pawns<const NOISY: bool, const QUIET: bool>(
    pos: &Position,
    list: &mut MoveList,
    us: Color,
    ksq: Square,
    check_mask: Bitboard,
) {
    let them = !us;
    let occ = pos.occupied();
    let empty = !occ;
    let theirs = pos.colored(them);
    let pinned = pos.pinned();
    let pawns = pos.pieces_of(us, PieceType::Pawn);

    let (up, rank7, rank3): (i8, _, _) = match us {
        Color::White => (8, Bitboard::rank(6), Bitboard::rank(2)),
        Color::Black => (-8, Bitboard::rank(1), Bitboard::rank(5)),
    };
    // A pinned pawn may only move along the pin line.
    let can_move = |from: Square, to: Square| !pinned.contains(from) || line(ksq, from).contains(to);

    let promoting = pawns & rank7;
    let normal = pawns & !rank7;

    if QUIET {
        let single = normal.forward(us) & empty;
        let double = (single & rank3).forward(us) & empty;
        for to in single & check_mask {
            let from = to.offset(-up);
            if can_move(from, to) {
                list.push(Move::new(from, to, Move::QUIET));
            }
        }
        for to in double & check_mask {
            let from = to.offset(-2 * up);
            if can_move(from, to) {
                list.push(Move::new(from, to, Move::DOUBLE_PUSH));
            }
        }
    }

    if NOISY {
        for from in normal {
            for to in pawn_attacks(us, from) & theirs & check_mask {
                if can_move(from, to) {
                    list.push(Move::new(from, to, Move::CAPTURE));
                }
            }
        }

        if let Some(ep) = pos.ep_square() {
            let cap_sq = Square::new(ep.raw() ^ 8);
            for from in pawn_attacks(them, ep) & normal {
                // Rare and full of edge cases (rank pins, discovered checks):
                // just verify the king is safe on the resulting occupancy.
                let occ_after = occ ^ from.bb() ^ ep.bb() ^ cap_sq.bb();
                let attackers = pos.attackers_to(ksq, occ_after) & theirs & !cap_sq.bb();
                if attackers.is_empty() {
                    list.push(Move::new(from, ep, Move::EN_PASSANT));
                }
            }
        }
    }

    for from in promoting {
        let to = from.offset(up);
        if empty.contains(to) && check_mask.contains(to) && can_move(from, to) {
            if NOISY {
                list.push(Move::promotion(from, to, PieceType::Queen, false));
            }
            if QUIET {
                for pt in [PieceType::Knight, PieceType::Rook, PieceType::Bishop] {
                    list.push(Move::promotion(from, to, pt, false));
                }
            }
        }
        if NOISY {
            for to in pawn_attacks(us, from) & theirs & check_mask {
                if can_move(from, to) {
                    for pt in [PieceType::Queen, PieceType::Knight, PieceType::Rook, PieceType::Bishop] {
                        list.push(Move::promotion(from, to, pt, true));
                    }
                }
            }
        }
    }
}

/// Chess960-aware castling. The king ends on g/c, the rook on f/d; every
/// square either piece crosses must be empty (ignoring the two movers), and
/// no square the king crosses may be attacked.
fn generate_castling(pos: &Position, list: &mut MoveList, us: Color, ksq: Square) {
    let occ = pos.occupied();
    for side in 0..2 {
        let idx = castle_index(us, side);
        if pos.castling_rights() & (1 << idx) == 0 {
            continue;
        }
        let rook_sq = pos.castle_rook(idx);
        let rank = ksq.rank();
        let king_dest = Square::from_coords(if side == 0 { 6 } else { 2 }, rank);
        let rook_dest = Square::from_coords(if side == 0 { 5 } else { 3 }, rank);

        let movers = ksq.bb() | rook_sq.bb();
        let king_path = between(ksq, king_dest) | king_dest.bb();
        let must_be_empty = (king_path | between(rook_sq, rook_dest) | rook_dest.bb()) & !movers;
        if (occ & must_be_empty).any() {
            continue;
        }
        // Lift both movers so a rook shielding the king along the back rank
        // is accounted for.
        let occ_after = occ ^ movers;
        let mut path = king_path;
        if Iterator::any(&mut path, |sq| pos.is_attacked_by(sq, !us, occ_after)) {
            continue;
        }
        list.push(Move::new(ksq, rook_sq, Move::CASTLE));
    }
}

/// Whether an arbitrary move (e.g. a TT move, possibly from a hash collision)
/// is legal in `pos`. Agrees exactly with `generate_all`; cheap for ordinary
/// moves, and defers the rare castling and en passant cases to the generator.
pub fn is_legal(pos: &Position, mv: Move) -> bool {
    use PieceType::*;
    if mv.is_null() {
        return false;
    }
    let us = pos.side_to_move();
    let them = !us;
    let from = mv.from();
    let to = mv.to();
    let piece = pos.piece_on(from);
    if piece.is_none() || piece.color() != us {
        return false;
    }
    if mv.is_castle() || mv.is_en_passant() {
        return legal_moves(pos).contains(&mv);
    }

    let target = pos.piece_on(to);
    let flag = mv.flag();
    if mv.is_capture() {
        if target.is_none() || target.color() != them || target.piece_type() == King {
            return false;
        }
    } else if !target.is_none() {
        return false;
    }

    let occ = pos.occupied();
    let pt = piece.piece_type();
    if pt == Pawn {
        let (up, start_rank, last_rank): (i8, u8, u8) = if us == Color::White { (8, 1, 7) } else { (-8, 6, 0) };
        if (to.rank() == last_rank) != mv.is_promotion() {
            return false;
        }
        let ok = if mv.is_capture() {
            pawn_attacks(us, from).contains(to) && (flag == Move::CAPTURE || flag >= Move::PROMO_CAPTURE)
        } else if flag == Move::DOUBLE_PUSH {
            from.rank() == start_rank && to == from.offset(2 * up) && !occ.contains(from.offset(up))
        } else {
            to == from.offset(up) && (flag == Move::QUIET || (Move::PROMO..Move::PROMO_CAPTURE).contains(&flag))
        };
        if !ok {
            return false;
        }
    } else {
        let expected = if mv.is_capture() { Move::CAPTURE } else { Move::QUIET };
        if flag != expected {
            return false;
        }
        let attacks = match pt {
            Knight => knight_attacks(from),
            Bishop => bishop_attacks(from, occ),
            Rook => rook_attacks(from, occ),
            Queen => queen_attacks(from, occ),
            _ => king_attacks(from),
        };
        if !attacks.contains(to) {
            return false;
        }
    }

    // King safety on the resulting occupancy; a captured piece no longer attacks.
    let ksq = if pt == King { to } else { pos.king_sq(us) };
    let occ_after = (occ ^ from.bb()) | to.bb();
    (pos.attackers_to(ksq, occ_after) & pos.colored(them) & !to.bb()).is_empty()
}
