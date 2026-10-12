//! Move ordering: the worker scores every move once, then the picker hands
//! them out lazily by selection sort (most nodes cut off after a move or two).

use crate::eval::PIECE_VALUES;
use crate::moves::{Move, MoveList};
use crate::position::Position;
use crate::types::PieceType;

pub const TT_MOVE: i32 = 1 << 30;
pub const GOOD_NOISY: i32 = 1 << 28;
pub const KILLER_0: i32 = 1 << 27;
pub const KILLER_1: i32 = KILLER_0 - 1;
pub const BAD_NOISY: i32 = -(1 << 28);
pub const UNDER_PROMO: i32 = -(1 << 29);

/// Marks "no move" (root, or after a null move) in continuation keys.
pub const NO_KEY: usize = usize::MAX;

#[inline(always)]
pub fn history_index(mv: Move) -> usize {
    mv.from().idx() * 64 + mv.to().idx()
}

/// Piece-to key of `mv` played in `pos` (before the move).
#[inline(always)]
pub fn piece_to_key(pos: &Position, mv: Move) -> usize {
    pos.piece_on(mv.from()).idx() * 64 + mv.to().idx()
}

/// Piece type captured by `mv` (pawn for en passant).
#[inline(always)]
pub fn captured_type(pos: &Position, mv: Move) -> PieceType {
    if mv.is_en_passant() {
        PieceType::Pawn
    } else {
        let victim = pos.piece_on(mv.to());
        if victim.is_none() { PieceType::Pawn } else { victim.piece_type() }
    }
}

/// Gravity update: moves `entry` towards ±max by `bonus`, slowing near the cap.
#[inline(always)]
pub fn apply_bonus(entry: &mut i16, bonus: i32, max: i32) {
    let e = *entry as i32;
    *entry = (e + bonus - e * bonus.abs() / max) as i16;
}

pub struct MovePicker {
    moves: MoveList,
    scores: [i32; 256],
    next: usize,
}

impl MovePicker {
    pub fn new(moves: MoveList, scores: [i32; 256]) -> MovePicker {
        MovePicker { moves, scores, next: 0 }
    }

    pub fn len(&self) -> usize {
        self.moves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }

    /// Next best move and its ordering score.
    pub fn next(&mut self) -> Option<(Move, i32)> {
        let n = self.moves.len();
        if self.next >= n {
            return None;
        }
        let mut best = self.next;
        for i in self.next + 1..n {
            if self.scores[i] > self.scores[best] {
                best = i;
            }
        }
        self.scores.swap(self.next, best);
        let mv = self.moves.swap_get(self.next, best);
        let score = self.scores[self.next];
        self.next += 1;
        Some((mv, score))
    }
}

/// Captures and queen promotions — the moves searched in quiescence.
#[inline(always)]
pub fn is_noisy(mv: Move) -> bool {
    mv.is_capture() || mv.promo_piece() == Some(PieceType::Queen)
}

/// Material gained by a noisy move before any recapture.
#[inline(always)]
pub fn noisy_gain(pos: &Position, mv: Move) -> i32 {
    let victim = if mv.is_capture() { PIECE_VALUES[captured_type(pos, mv).idx()] } else { 0 };
    let promo = mv.promo_piece().map_or(0, |pt| PIECE_VALUES[pt.idx()] - PIECE_VALUES[PieceType::Pawn.idx()]);
    victim + promo
}
