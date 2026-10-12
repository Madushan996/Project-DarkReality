use std::fmt;
use std::ops::Not;

use crate::bitboard::Bitboard;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub const ALL: [Color; 2] = [Color::White, Color::Black];

    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }
}

impl Not for Color {
    type Output = Color;

    #[inline(always)]
    fn not(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl PieceType {
    pub const ALL: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];

    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }

    pub const fn to_char(self) -> char {
        b"pnbrqk"[self as usize] as char
    }
}

/// A coloured piece, encoded as `(piece_type << 1) | color`; 12 means empty.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Piece(u8);

impl Piece {
    pub const NONE: Piece = Piece(12);

    #[inline(always)]
    pub const fn new(color: Color, pt: PieceType) -> Piece {
        Piece(((pt as u8) << 1) | color as u8)
    }

    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 == 12
    }

    #[inline(always)]
    pub const fn color(self) -> Color {
        if self.0 & 1 == 0 { Color::White } else { Color::Black }
    }

    #[inline(always)]
    pub const fn piece_type(self) -> PieceType {
        PieceType::ALL[(self.0 >> 1) as usize]
    }

    #[inline(always)]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }

    pub fn to_char(self) -> char {
        if self.is_none() {
            return '.';
        }
        let c = self.piece_type().to_char();
        if self.color() == Color::White { c.to_ascii_uppercase() } else { c }
    }

    pub fn from_char(ch: char) -> Option<Piece> {
        let pt = match ch.to_ascii_lowercase() {
            'p' => PieceType::Pawn,
            'n' => PieceType::Knight,
            'b' => PieceType::Bishop,
            'r' => PieceType::Rook,
            'q' => PieceType::Queen,
            'k' => PieceType::King,
            _ => return None,
        };
        let color = if ch.is_ascii_uppercase() { Color::White } else { Color::Black };
        Some(Piece::new(color, pt))
    }
}

/// Square index, a1 = 0 … h8 = 63.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Square(u8);

impl Square {
    #[inline(always)]
    pub const fn new(idx: u8) -> Square {
        debug_assert!(idx < 64);
        Square(idx)
    }

    #[inline(always)]
    pub const fn from_coords(file: u8, rank: u8) -> Square {
        Square::new(rank * 8 + file)
    }

    #[inline(always)]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }

    #[inline(always)]
    pub const fn raw(self) -> u8 {
        self.0
    }

    #[inline(always)]
    pub const fn file(self) -> u8 {
        self.0 & 7
    }

    #[inline(always)]
    pub const fn rank(self) -> u8 {
        self.0 >> 3
    }

    #[inline(always)]
    pub const fn bb(self) -> Bitboard {
        Bitboard(1u64 << self.0)
    }

    /// Vertical mirror (a1 <-> a8).
    #[inline(always)]
    pub const fn flip(self) -> Square {
        Square(self.0 ^ 56)
    }

    #[inline(always)]
    pub const fn offset(self, delta: i8) -> Square {
        Square::new((self.0 as i8 + delta) as u8)
    }

    pub fn parse(s: &str) -> Option<Square> {
        let b = s.as_bytes();
        if b.len() != 2 || !(b'a'..=b'h').contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) {
            return None;
        }
        Some(Square::from_coords(b[0] - b'a', b[1] - b'1'))
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", (b'a' + self.file()) as char, (b'1' + self.rank()) as char)
    }
}
