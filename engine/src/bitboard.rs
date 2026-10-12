use std::fmt;
use std::ops::Not;

use crate::types::{Color, Square};

#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard(0);
    pub const ALL: Bitboard = Bitboard(!0);

    #[inline(always)]
    pub const fn rank(r: u8) -> Bitboard {
        Bitboard(0xFF << (8 * r))
    }

    #[inline(always)]
    pub const fn file(f: u8) -> Bitboard {
        Bitboard(0x0101_0101_0101_0101 << f)
    }

    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn any(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    #[inline(always)]
    pub const fn more_than_one(self) -> bool {
        self.0 & self.0.wrapping_sub(1) != 0
    }

    #[inline(always)]
    pub const fn contains(self, sq: Square) -> bool {
        self.0 & (1u64 << sq.raw()) != 0
    }

    /// Least significant square. The bitboard must not be empty.
    #[inline(always)]
    pub const fn lsb(self) -> Square {
        debug_assert!(self.0 != 0);
        Square::new(self.0.trailing_zeros() as u8)
    }

    /// Shift one rank towards the opponent of `color`.
    #[inline(always)]
    pub const fn forward(self, color: Color) -> Bitboard {
        match color {
            Color::White => Bitboard(self.0 << 8),
            Color::Black => Bitboard(self.0 >> 8),
        }
    }
}

macro_rules! impl_bit_ops {
    ($($tr:ident $f:ident $tra:ident $fa:ident $op:tt),*) => {$(
        impl std::ops::$tr for Bitboard {
            type Output = Bitboard;

            #[inline(always)]
            fn $f(self, rhs: Bitboard) -> Bitboard {
                Bitboard(self.0 $op rhs.0)
            }
        }

        impl std::ops::$tra for Bitboard {
            #[inline(always)]
            fn $fa(&mut self, rhs: Bitboard) {
                self.0 = self.0 $op rhs.0;
            }
        }
    )*};
}

impl_bit_ops!(
    BitAnd bitand BitAndAssign bitand_assign &,
    BitOr bitor BitOrAssign bitor_assign |,
    BitXor bitxor BitXorAssign bitxor_assign ^
);

impl Not for Bitboard {
    type Output = Bitboard;

    #[inline(always)]
    fn not(self) -> Bitboard {
        Bitboard(!self.0)
    }
}

impl Iterator for Bitboard {
    type Item = Square;

    #[inline(always)]
    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 {
            return None;
        }
        let sq = Square::new(self.0.trailing_zeros() as u8);
        self.0 &= self.0 - 1;
        Some(sq)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.count_ones() as usize;
        (n, Some(n))
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Bitboard({:#018x})", self.0)?;
        for rank in (0..8).rev() {
            for file in 0..8 {
                let c = if self.contains(Square::from_coords(file, rank)) { 'X' } else { '.' };
                write!(f, "{c} ")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
