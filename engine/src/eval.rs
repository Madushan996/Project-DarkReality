//! Gen0 evaluation: material only.
//!
//! This is deliberately knowledge-free. It is the root of DarkReality's
//! network lineage: self-play games with this eval
//! produce the data for the first NNUE, so no foreign evaluation knowledge
//! enters the engine.

use crate::position::Position;
use crate::types::PieceType;

pub const PIECE_VALUES: [i32; 6] = [100, 300, 300, 500, 900, 0];

/// Static evaluation in centipawns from the side to move's point of view.
pub fn evaluate(pos: &Position) -> i32 {
    let us = pos.side_to_move();
    let them = !us;
    let mut score = 0;
    for pt in [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
        let diff = pos.pieces_of(us, pt).count() as i32 - pos.pieces_of(them, pt).count() as i32;
        score += PIECE_VALUES[pt.idx()] * diff;
    }
    score
}
