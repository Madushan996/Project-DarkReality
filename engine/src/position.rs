use std::fmt;

use crate::attacks::*;
use crate::bitboard::Bitboard;
use crate::moves::Move;
use crate::types::{Color, Piece, PieceType, Square};
use crate::zobrist::KEYS;

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Castling-right bit index: `color * 2 + side`, side 0 = kingside (h-side),
/// side 1 = queenside (a-side).
#[inline(always)]
pub const fn castle_index(color: Color, side: usize) -> usize {
    color.idx() * 2 + side
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FenError(pub String);

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invalid FEN: {}", self.0)
    }
}

impl std::error::Error for FenError {}

fn fen_err<T>(msg: impl Into<String>) -> Result<T, FenError> {
    Err(FenError(msg.into()))
}

/// Full board state. Copy-make: `make_move` returns a new position.
///
/// No `PartialEq`: `castle_rooks` keeps stale squares for lost rights, so
/// compare positions by FEN or hash instead.
#[derive(Clone, Copy)]
pub struct Position {
    pieces: [Bitboard; 6],
    colors: [Bitboard; 2],
    board: [Piece; 64],
    stm: Color,
    castling: u8,
    castle_rooks: [Square; 4],
    ep: Option<Square>,
    halfmove: u16,
    fullmove: u16,
    hash: u64,
    /// Zobrist hash of the pawns only (for pawn-structure correction history).
    pawn_hash: u64,
    /// Zobrist hash of each colour's non-pawn pieces (kings included).
    nonpawn_hash: [u64; 2],
    checkers: Bitboard,
    pinned: Bitboard,
}

impl Position {
    fn empty() -> Position {
        Position {
            pieces: [Bitboard::EMPTY; 6],
            colors: [Bitboard::EMPTY; 2],
            board: [Piece::NONE; 64],
            stm: Color::White,
            castling: 0,
            castle_rooks: [Square::new(0); 4],
            ep: None,
            halfmove: 0,
            fullmove: 1,
            hash: 0,
            pawn_hash: 0,
            nonpawn_hash: [0; 2],
            checkers: Bitboard::EMPTY,
            pinned: Bitboard::EMPTY,
        }
    }

    pub fn startpos() -> Position {
        Position::from_fen(START_FEN).expect("start FEN is valid")
    }

    /// Parses standard FEN, X-FEN (KQkq = outermost rook) and Shredder-FEN
    /// (rook files, e.g. `HAha`) castling fields.
    pub fn from_fen(fen: &str) -> Result<Position, FenError> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        if fields.len() < 4 {
            return fen_err("expected at least 4 fields");
        }
        let mut pos = Position::empty();

        let ranks: Vec<&str> = fields[0].split('/').collect();
        if ranks.len() != 8 {
            return fen_err("board must have 8 ranks");
        }
        for (i, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - i as u8;
            let mut file = 0u8;
            for ch in rank_str.chars() {
                if let Some(d) = ch.to_digit(10) {
                    file += d as u8;
                } else {
                    let Some(piece) = Piece::from_char(ch) else {
                        return fen_err(format!("bad piece '{ch}'"));
                    };
                    if file >= 8 {
                        return fen_err("rank overflow");
                    }
                    pos.add_piece(Square::from_coords(file, rank), piece);
                    file += 1;
                }
                if file > 8 {
                    return fen_err("rank overflow");
                }
            }
            if file != 8 {
                return fen_err(format!("rank {} has {file} files", rank + 1));
            }
        }
        for color in Color::ALL {
            if pos.pieces_of(color, PieceType::King).count() != 1 {
                return fen_err("each side needs exactly one king");
            }
        }
        if (pos.pieces(PieceType::Pawn) & (Bitboard::rank(0) | Bitboard::rank(7))).any() {
            return fen_err("pawn on back rank");
        }

        pos.stm = match fields[1] {
            "w" => Color::White,
            "b" => Color::Black,
            s => return fen_err(format!("bad side to move '{s}'")),
        };

        if fields[2] != "-" {
            for ch in fields[2].chars() {
                pos.parse_castling_char(ch)?;
            }
        }

        if fields[3] != "-" {
            let Some(ep) = Square::parse(fields[3]) else {
                return fen_err(format!("bad en passant square '{}'", fields[3]));
            };
            // Only keep it if a capture is pseudo-legally possible, matching make_move.
            if (pawn_attacks(!pos.stm, ep) & pos.pieces_of(pos.stm, PieceType::Pawn)).any() {
                pos.ep = Some(ep);
            }
        }

        pos.halfmove = fields.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        pos.fullmove = fields.get(5).and_then(|s| s.parse().ok()).unwrap_or(1).max(1);

        let them = !pos.stm;
        if pos.is_attacked_by(pos.king_sq(them), pos.stm, pos.occupied()) {
            return fen_err("side not to move is in check");
        }

        pos.hash = pos.compute_hash();
        pos.update_check_info();
        Ok(pos)
    }

    fn parse_castling_char(&mut self, ch: char) -> Result<(), FenError> {
        let color = if ch.is_ascii_uppercase() { Color::White } else { Color::Black };
        let back_rank = if color == Color::White { 0 } else { 7 };
        let ksq = self.king_sq(color);
        if ksq.rank() != back_rank {
            return fen_err(format!("castling right '{ch}' but king not on back rank"));
        }
        let rook = Piece::new(color, PieceType::Rook);
        let is_rook = |file: u8| self.board[Square::from_coords(file, back_rank).idx()] == rook;

        let rook_file = match ch.to_ascii_lowercase() {
            'k' => (ksq.file() + 1..8).rev().find(|&f| is_rook(f)),
            'q' => (0..ksq.file()).find(|&f| is_rook(f)),
            c @ 'a'..='h' => Some(c as u8 - b'a').filter(|&f| is_rook(f)),
            _ => return fen_err(format!("bad castling char '{ch}'")),
        };
        let Some(rook_file) = rook_file else {
            return fen_err(format!("castling right '{ch}' has no rook"));
        };
        let side = if rook_file > ksq.file() { 0 } else { 1 };
        let idx = castle_index(color, side);
        self.castling |= 1 << idx;
        self.castle_rooks[idx] = Square::from_coords(rook_file, back_rank);
        Ok(())
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let piece = self.board[Square::from_coords(file, rank).idx()];
                if piece.is_none() {
                    empty += 1;
                } else {
                    if empty > 0 {
                        fen.push_str(&empty.to_string());
                        empty = 0;
                    }
                    fen.push(piece.to_char());
                }
            }
            if empty > 0 {
                fen.push_str(&empty.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }
        fen.push_str(if self.stm == Color::White { " w " } else { " b " });

        if self.castling == 0 {
            fen.push('-');
        }
        for idx in 0..4 {
            if self.castling & (1 << idx) == 0 {
                continue;
            }
            let color = if idx < 2 { Color::White } else { Color::Black };
            let rook_sq = self.castle_rooks[idx];
            let ksq = self.king_sq(color);
            // Use K/Q when the castling rook is the outermost one on its side.
            let edge = Square::from_coords(if rook_sq > ksq { 7 } else { 0 }, rook_sq.rank());
            let outer = (between(rook_sq, edge) | edge.bb()) & !rook_sq.bb();
            let ch = if (outer & self.pieces_of(color, PieceType::Rook)).is_empty() {
                if idx % 2 == 0 { 'k' } else { 'q' }
            } else {
                (b'a' + rook_sq.file()) as char
            };
            fen.push(if color == Color::White { ch.to_ascii_uppercase() } else { ch });
        }

        match self.ep {
            Some(sq) => fen.push_str(&format!(" {sq}")),
            None => fen.push_str(" -"),
        }
        fen.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        fen
    }

    // ---------------------------------------------------------------- accessors

    #[inline(always)]
    pub fn side_to_move(&self) -> Color {
        self.stm
    }

    #[inline(always)]
    pub fn piece_on(&self, sq: Square) -> Piece {
        self.board[sq.idx()]
    }

    #[inline(always)]
    pub fn pieces(&self, pt: PieceType) -> Bitboard {
        self.pieces[pt.idx()]
    }

    #[inline(always)]
    pub fn colored(&self, color: Color) -> Bitboard {
        self.colors[color.idx()]
    }

    #[inline(always)]
    pub fn pieces_of(&self, color: Color, pt: PieceType) -> Bitboard {
        self.pieces[pt.idx()] & self.colors[color.idx()]
    }

    #[inline(always)]
    pub fn occupied(&self) -> Bitboard {
        self.colors[0] | self.colors[1]
    }

    #[inline(always)]
    pub fn king_sq(&self, color: Color) -> Square {
        self.pieces_of(color, PieceType::King).lsb()
    }

    #[inline(always)]
    pub fn checkers(&self) -> Bitboard {
        self.checkers
    }

    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.checkers.any()
    }

    /// Side-to-move pieces pinned to their own king.
    #[inline(always)]
    pub fn pinned(&self) -> Bitboard {
        self.pinned
    }

    #[inline(always)]
    pub fn hash(&self) -> u64 {
        self.hash
    }

    #[inline(always)]
    pub fn pawn_hash(&self) -> u64 {
        self.pawn_hash
    }

    #[inline(always)]
    pub fn nonpawn_hash(&self, color: Color) -> u64 {
        self.nonpawn_hash[color.idx()]
    }

    #[inline(always)]
    pub fn ep_square(&self) -> Option<Square> {
        self.ep
    }

    #[inline(always)]
    pub fn castling_rights(&self) -> u8 {
        self.castling
    }

    #[inline(always)]
    pub fn castle_rook(&self, idx: usize) -> Square {
        self.castle_rooks[idx]
    }

    #[inline(always)]
    pub fn halfmove_clock(&self) -> u16 {
        self.halfmove
    }

    #[inline(always)]
    pub fn fullmove_number(&self) -> u16 {
        self.fullmove
    }

    /// Every square the side not to move attacks (pins and legality ignored).
    pub fn threats(&self) -> Bitboard {
        use PieceType::*;
        let them = !self.stm;
        let occ = self.occupied();
        let mut t = Bitboard::EMPTY;
        for sq in self.pieces_of(them, Pawn) {
            t |= pawn_attacks(them, sq);
        }
        for sq in self.pieces_of(them, Knight) {
            t |= knight_attacks(sq);
        }
        for sq in self.pieces_of(them, Bishop) | self.pieces_of(them, Queen) {
            t |= bishop_attacks(sq, occ);
        }
        for sq in self.pieces_of(them, Rook) | self.pieces_of(them, Queen) {
            t |= rook_attacks(sq, occ);
        }
        t | king_attacks(self.king_sq(them))
    }

    /// Whether `color` has any piece besides pawns and king (null-move safety).
    #[inline]
    pub fn has_non_pawn_material(&self, color: Color) -> bool {
        (self.colored(color) & !(self.pieces(PieceType::Pawn) | self.pieces(PieceType::King))).any()
    }

    /// Dead draws by material: K vs K, K+minor vs K.
    pub fn is_insufficient_material(&self) -> bool {
        use PieceType::*;
        let heavy = self.pieces(Pawn) | self.pieces(Rook) | self.pieces(Queen);
        heavy.is_empty() && (self.pieces(Knight) | self.pieces(Bishop)).count() <= 1
    }

    // ------------------------------------------------------------------ attacks

    /// All pieces of both colours attacking `sq`, with `occ` as blockers.
    #[inline]
    pub fn attackers_to(&self, sq: Square, occ: Bitboard) -> Bitboard {
        use PieceType::*;
        let rooks = self.pieces(Rook) | self.pieces(Queen);
        let bishops = self.pieces(Bishop) | self.pieces(Queen);
        (pawn_attacks(Color::White, sq) & self.pieces_of(Color::Black, Pawn))
            | (pawn_attacks(Color::Black, sq) & self.pieces_of(Color::White, Pawn))
            | (knight_attacks(sq) & self.pieces(Knight))
            | (king_attacks(sq) & self.pieces(King))
            | (bishop_attacks(sq, occ) & bishops)
            | (rook_attacks(sq, occ) & rooks)
    }

    #[inline]
    pub fn is_attacked_by(&self, sq: Square, by: Color, occ: Bitboard) -> bool {
        use PieceType::*;
        let them = self.colored(by);
        (pawn_attacks(!by, sq) & self.pieces(Pawn) & them).any()
            || (knight_attacks(sq) & self.pieces(Knight) & them).any()
            || (king_attacks(sq) & self.pieces(King) & them).any()
            || (bishop_attacks(sq, occ) & (self.pieces(Bishop) | self.pieces(Queen)) & them).any()
            || (rook_attacks(sq, occ) & (self.pieces(Rook) | self.pieces(Queen)) & them).any()
    }

    fn update_check_info(&mut self) {
        use PieceType::*;
        let us = self.stm;
        let them = !us;
        let ksq = self.king_sq(us);
        let occ = self.occupied();

        self.checkers = self.attackers_to(ksq, occ) & self.colored(them);

        let snipers = ((rook_attacks(ksq, Bitboard::EMPTY) & (self.pieces(Rook) | self.pieces(Queen)))
            | (bishop_attacks(ksq, Bitboard::EMPTY) & (self.pieces(Bishop) | self.pieces(Queen))))
            & self.colored(them);
        let mut pinned = Bitboard::EMPTY;
        for sniper in snipers {
            let blockers = between(ksq, sniper) & occ;
            if blockers.any() && !blockers.more_than_one() {
                pinned |= blockers;
            }
        }
        self.pinned = pinned & self.colored(us);
    }

    // -------------------------------------------------------------- make / hash

    #[inline(always)]
    fn add_piece(&mut self, sq: Square, piece: Piece) {
        debug_assert!(self.board[sq.idx()].is_none());
        let bb = sq.bb();
        self.board[sq.idx()] = piece;
        self.pieces[piece.piece_type().idx()] |= bb;
        self.colors[piece.color().idx()] |= bb;
        self.hash ^= KEYS.pieces[piece.idx()][sq.idx()];
        if piece.piece_type() == PieceType::Pawn {
            self.pawn_hash ^= KEYS.pieces[piece.idx()][sq.idx()];
        } else {
            self.nonpawn_hash[piece.color().idx()] ^= KEYS.pieces[piece.idx()][sq.idx()];
        }
    }

    #[inline(always)]
    fn remove_piece(&mut self, sq: Square) -> Piece {
        let piece = self.board[sq.idx()];
        debug_assert!(!piece.is_none());
        let bb = sq.bb();
        self.board[sq.idx()] = Piece::NONE;
        self.pieces[piece.piece_type().idx()] ^= bb;
        self.colors[piece.color().idx()] ^= bb;
        self.hash ^= KEYS.pieces[piece.idx()][sq.idx()];
        if piece.piece_type() == PieceType::Pawn {
            self.pawn_hash ^= KEYS.pieces[piece.idx()][sq.idx()];
        } else {
            self.nonpawn_hash[piece.color().idx()] ^= KEYS.pieces[piece.idx()][sq.idx()];
        }
        piece
    }

    /// Applies a legal move and returns the resulting position.
    pub fn make_move(&self, mv: Move) -> Position {
        let mut p = *self;
        let us = self.stm;
        let them = !us;
        let from = mv.from();
        let to = mv.to();
        let piece = self.board[from.idx()];
        debug_assert!(!piece.is_none() && piece.color() == us, "bad move {mv:?} in {}", self.to_fen());

        if let Some(ep) = p.ep.take() {
            p.hash ^= KEYS.ep_file[ep.file() as usize];
        }
        p.hash ^= KEYS.castling[p.castling as usize];
        p.halfmove += 1;
        if us == Color::Black {
            p.fullmove += 1;
        }

        if mv.is_castle() {
            let kingside = to.file() > from.file();
            let rank = from.rank();
            let king_dest = Square::from_coords(if kingside { 6 } else { 2 }, rank);
            let rook_dest = Square::from_coords(if kingside { 5 } else { 3 }, rank);
            // Remove both first: in Chess960 the destinations may overlap the origins.
            let king = p.remove_piece(from);
            let rook = p.remove_piece(to);
            p.add_piece(king_dest, king);
            p.add_piece(rook_dest, rook);
        } else {
            if mv.is_capture() {
                // En passant: the captured pawn sits one rank behind `to`.
                let cap_sq = if mv.is_en_passant() { Square::new(to.raw() ^ 8) } else { to };
                p.remove_piece(cap_sq);
                p.halfmove = 0;
            }
            p.remove_piece(from);
            let placed = match mv.promo_piece() {
                Some(pt) => Piece::new(us, pt),
                None => piece,
            };
            p.add_piece(to, placed);

            if piece.piece_type() == PieceType::Pawn {
                p.halfmove = 0;
                if mv.flag() == Move::DOUBLE_PUSH {
                    let ep = Square::new((from.raw() + to.raw()) / 2);
                    if (pawn_attacks(us, ep) & p.pieces_of(them, PieceType::Pawn)).any() {
                        p.ep = Some(ep);
                        p.hash ^= KEYS.ep_file[ep.file() as usize];
                    }
                }
            }
        }

        if p.castling != 0 {
            let mut lost = 0u8;
            if piece.piece_type() == PieceType::King {
                lost |= 0b11 << (us.idx() * 2);
            }
            for idx in 0..4 {
                let rook_sq = p.castle_rooks[idx];
                if rook_sq == from || rook_sq == to {
                    lost |= 1 << idx;
                }
            }
            p.castling &= !lost;
        }
        p.hash ^= KEYS.castling[p.castling as usize];

        p.stm = them;
        p.hash ^= KEYS.side;
        p.update_check_info();
        p
    }

    /// Passes the turn. Must not be called while in check.
    pub fn make_null(&self) -> Position {
        debug_assert!(!self.in_check());
        let mut p = *self;
        if let Some(ep) = p.ep.take() {
            p.hash ^= KEYS.ep_file[ep.file() as usize];
        }
        // Reset so repetition detection never matches across the null move.
        p.halfmove = 0;
        p.stm = !p.stm;
        p.hash ^= KEYS.side;
        p.update_check_info();
        p
    }

    /// Hash computed from scratch; used for initialisation and debugging.
    pub fn compute_hash(&self) -> u64 {
        let mut hash = 0;
        for sq in self.occupied() {
            hash ^= KEYS.pieces[self.board[sq.idx()].idx()][sq.idx()];
        }
        hash ^= KEYS.castling[self.castling as usize];
        if let Some(ep) = self.ep {
            hash ^= KEYS.ep_file[ep.file() as usize];
        }
        if self.stm == Color::Black {
            hash ^= KEYS.side;
        }
        hash
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "  +-----------------+")?;
        for rank in (0..8).rev() {
            write!(f, "{} | ", rank + 1)?;
            for file in 0..8 {
                write!(f, "{} ", self.board[Square::from_coords(file, rank).idx()].to_char())?;
            }
            writeln!(f, "|")?;
        }
        writeln!(f, "  +-----------------+")?;
        writeln!(f, "    a b c d e f g h")?;
        writeln!(f)?;
        writeln!(f, "Fen: {}", self.to_fen())?;
        writeln!(f, "Key: {:016x}", self.hash)?;
        let checkers: Vec<String> = self.checkers.map(|sq| sq.to_string()).collect();
        write!(f, "Checkers: {}", checkers.join(" "))
    }
}

impl fmt::Debug for Position {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Position({})", self.to_fen())
    }
}
