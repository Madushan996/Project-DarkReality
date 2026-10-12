//! Attack lookups. Slider tables and square relations come from `build.rs`.

use crate::bitboard::Bitboard;
use crate::types::{Color, Square};

pub struct MagicEntry {
    pub mask: u64,
    pub magic: u64,
    pub shift: u32,
    pub offset: u32,
}

include!(concat!(env!("OUT_DIR"), "/tables.rs"));

const NOT_A: u64 = 0xfefe_fefe_fefe_fefe;
const NOT_AB: u64 = 0xfcfc_fcfc_fcfc_fcfc;
const NOT_H: u64 = 0x7f7f_7f7f_7f7f_7f7f;
const NOT_GH: u64 = 0x3f3f_3f3f_3f3f_3f3f;

static KNIGHT: [u64; 64] = {
    let mut table = [0; 64];
    let mut sq = 0;
    while sq < 64 {
        let b = 1u64 << sq;
        table[sq] = ((b << 17) & NOT_A)
            | ((b << 15) & NOT_H)
            | ((b << 10) & NOT_AB)
            | ((b << 6) & NOT_GH)
            | ((b >> 17) & NOT_H)
            | ((b >> 15) & NOT_A)
            | ((b >> 10) & NOT_GH)
            | ((b >> 6) & NOT_AB);
        sq += 1;
    }
    table
};

static KING: [u64; 64] = {
    let mut table = [0; 64];
    let mut sq = 0;
    while sq < 64 {
        let b = 1u64 << sq;
        table[sq] = (b << 8)
            | (b >> 8)
            | (((b << 1) | (b << 9) | (b >> 7)) & NOT_A)
            | (((b >> 1) | (b >> 9) | (b << 7)) & NOT_H);
        sq += 1;
    }
    table
};

static PAWN: [[u64; 64]; 2] = {
    let mut table = [[0; 64]; 2];
    let mut sq = 0;
    while sq < 64 {
        let b = 1u64 << sq;
        table[0][sq] = ((b << 9) & NOT_A) | ((b << 7) & NOT_H);
        table[1][sq] = ((b >> 7) & NOT_A) | ((b >> 9) & NOT_H);
        sq += 1;
    }
    table
};

#[inline(always)]
pub fn knight_attacks(sq: Square) -> Bitboard {
    Bitboard(KNIGHT[sq.idx()])
}

#[inline(always)]
pub fn king_attacks(sq: Square) -> Bitboard {
    Bitboard(KING[sq.idx()])
}

/// Squares attacked by a pawn of `color` standing on `sq`.
#[inline(always)]
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    Bitboard(PAWN[color.idx()][sq.idx()])
}

#[inline(always)]
fn magic_lookup(entry: &MagicEntry, occ: Bitboard) -> Bitboard {
    let idx = ((occ.0 & entry.mask).wrapping_mul(entry.magic) >> entry.shift) as usize;
    Bitboard(SLIDER_TABLE[entry.offset as usize + idx])
}

#[inline(always)]
pub fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    magic_lookup(&ROOK_MAGICS[sq.idx()], occ)
}

#[inline(always)]
pub fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    magic_lookup(&BISHOP_MAGICS[sq.idx()], occ)
}

#[inline(always)]
pub fn queen_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    rook_attacks(sq, occ) | bishop_attacks(sq, occ)
}

/// Squares strictly between `a` and `b` if they share a rank, file or diagonal.
#[inline(always)]
pub fn between(a: Square, b: Square) -> Bitboard {
    Bitboard(BETWEEN[a.idx()][b.idx()])
}

/// The full line through `a` and `b` if aligned, otherwise empty.
#[inline(always)]
pub fn line(a: Square, b: Square) -> Bitboard {
    Bitboard(LINE[a.idx()][b.idx()])
}
