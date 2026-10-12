use std::fmt;
use std::ops::Deref;

use crate::types::{PieceType, Square};

/// 16-bit move: from (6) | to (6) | flags (4).
///
/// Flags: bit 2 = capture, bit 3 = promotion (low 2 bits = N/B/R/Q).
/// Castling is encoded as king-captures-own-rook (`to` = rook square), which
/// supports both standard chess and Chess960 with one code path.
#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Move(u16);

impl Move {
    pub const NULL: Move = Move(0);

    pub const QUIET: u16 = 0;
    pub const DOUBLE_PUSH: u16 = 1;
    pub const CASTLE: u16 = 2;
    pub const CAPTURE: u16 = 4;
    pub const EN_PASSANT: u16 = 5;
    pub const PROMO: u16 = 8;
    pub const PROMO_CAPTURE: u16 = 12;

    #[inline(always)]
    pub const fn new(from: Square, to: Square, flag: u16) -> Move {
        Move(from.raw() as u16 | (to.raw() as u16) << 6 | flag << 12)
    }

    /// Rebuilds a move from `raw()`. The caller must validate it against the
    /// position (e.g. by matching generated moves) before playing it.
    #[inline(always)]
    pub const fn from_raw(raw: u16) -> Move {
        Move(raw)
    }

    #[inline(always)]
    pub const fn promotion(from: Square, to: Square, pt: PieceType, capture: bool) -> Move {
        let base = if capture { Move::PROMO_CAPTURE } else { Move::PROMO };
        Move::new(from, to, base | (pt as u16 - 1))
    }

    #[inline(always)]
    pub const fn from(self) -> Square {
        Square::new((self.0 & 63) as u8)
    }

    #[inline(always)]
    pub const fn to(self) -> Square {
        Square::new(((self.0 >> 6) & 63) as u8)
    }

    #[inline(always)]
    pub const fn flag(self) -> u16 {
        self.0 >> 12
    }

    #[inline(always)]
    pub const fn raw(self) -> u16 {
        self.0
    }

    #[inline(always)]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn is_capture(self) -> bool {
        self.flag() & Move::CAPTURE != 0
    }

    #[inline(always)]
    pub const fn is_promotion(self) -> bool {
        self.flag() & Move::PROMO != 0
    }

    #[inline(always)]
    pub const fn is_castle(self) -> bool {
        self.flag() == Move::CASTLE
    }

    #[inline(always)]
    pub const fn is_en_passant(self) -> bool {
        self.flag() == Move::EN_PASSANT
    }

    #[inline(always)]
    pub const fn promo_piece(self) -> Option<PieceType> {
        if self.is_promotion() {
            Some(PieceType::ALL[(self.flag() & 3) as usize + 1])
        } else {
            None
        }
    }

    /// For castling, the square the king lands on.
    pub const fn castle_king_dest(self) -> Square {
        let file = if self.to().file() > self.from().file() { 6 } else { 2 };
        Square::from_coords(file, self.from().rank())
    }

    /// UCI notation. Standard chess writes castling as the king's two-square
    /// move; Chess960 mode writes it as king-takes-rook.
    pub fn to_uci(self, chess960: bool) -> String {
        if self.is_null() {
            return "0000".to_string();
        }
        let to = if self.is_castle() && !chess960 { self.castle_king_dest() } else { self.to() };
        match self.promo_piece() {
            Some(pt) => format!("{}{}{}", self.from(), to, pt.to_char()),
            None => format!("{}{}", self.from(), to),
        }
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.to_uci(false))
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Move({}, flag {})", self.to_uci(true), self.flag())
    }
}

/// Fixed-capacity move buffer (no position has more than 218 legal moves).
pub struct MoveList {
    moves: [Move; 256],
    len: usize,
}

impl MoveList {
    #[inline(always)]
    pub fn new() -> MoveList {
        MoveList { moves: [Move::NULL; 256], len: 0 }
    }

    #[inline(always)]
    pub fn push(&mut self, mv: Move) {
        self.moves[self.len] = mv;
        self.len += 1;
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Swaps entries `a` and `b`, returning the move now at `a`.
    #[inline(always)]
    pub fn swap_get(&mut self, a: usize, b: usize) -> Move {
        self.moves.swap(a, b);
        self.moves[a]
    }
}

impl Default for MoveList {
    fn default() -> Self {
        MoveList::new()
    }
}

impl Deref for MoveList {
    type Target = [Move];

    #[inline(always)]
    fn deref(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}
