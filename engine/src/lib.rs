pub mod attacks;
pub mod bench;
pub mod bitboard;
pub mod datagen;
pub mod eval;
pub mod movegen;
pub mod moves;
pub mod nnue;
pub mod perft;
pub mod position;
pub mod search;
pub mod see;
pub mod types;
pub mod uci;
pub mod zobrist;

use position::Position;

/// Static evaluation of `pos` from the side to move's view, using the
/// embedded network if there is one: (centipawns, raw network units).
pub fn static_eval(pos: &Position) -> (i32, Option<i32>) {
    match nnue::network() {
        Some(net) => {
            let mut acc = nnue::Accumulator::default();
            acc.refresh(net, pos);
            let pieces = pos.occupied().count();
            (net.evaluate(&acc, pos.side_to_move(), pieces), Some(net.evaluate_units(&acc, pos.side_to_move(), pieces)))
        }
        None => (eval::evaluate(pos), None),
    }
}
