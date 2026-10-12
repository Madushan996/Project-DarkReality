//! Original, bounded-memory converter using only our own engine's rules/packing.
//! stdin: FEN<TAB>white-centipawns[<TAB>quiet-PV-first-move].
//! stdout: DRP records; stderr: JSON statistics. No outcome is fabricated.
use std::io::{self, BufRead, BufWriter, Write};
use darkreality::{datagen::format::{pack_position, GameResult}, movegen::legal_moves,
                 position::Position, types::{Color, PieceType}};

fn convert(line: &str) -> Option<[u8; 32]> {
    let mut fields = line.trim_end().split('\t');
    let fen = fields.next()?;
    let cp: i16 = fields.next()?.parse().ok()?;
    if cp.unsigned_abs() > 1500 { return None; }
    let pv = fields.next();
    if fields.next().is_some() { return None; }
    let pos = Position::from_fen(fen).ok()?;
    let count = pos.occupied().count();
    if !(6..=32).contains(&count) || pos.in_check() { return None; }
    for color in Color::ALL {
        if pos.pieces_of(color, PieceType::Pawn).count() > 8 || pos.colored(color).count() > 16 {
            return None;
        }
    }
    let moves = legal_moves(&pos);
    if moves.is_empty() { return None; }
    if let Some(pv) = pv {
        let mv = moves.iter().find(|mv| mv.to_uci(true) == pv)?;
        if mv.is_capture() || mv.is_promotion() { return None; }
    }
    let mut rec = pack_position(&pos, cp, GameResult::Draw);
    rec[30] = 1; // score-only, placeholder result must never train as a draw
    Some(rec)
}

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let mut output = BufWriter::with_capacity(1 << 20, io::stdout().lock());
    let mut line = String::new();
    let mut input = stdin.lock();
    let (mut seen, mut accepted) = (0u64, 0u64);
    let start = std::time::Instant::now();
    while input.read_line(&mut line)? != 0 {
        seen += 1;
        if let Some(rec) = convert(&line) {
            output.write_all(&rec)?;
            accepted += 1;
        }
        line.clear();
    }
    output.flush()?;
    eprintln!("{{\"seen\":{seen},\"accepted\":{accepted},\"seconds\":{}}}", start.elapsed().as_secs_f64());
    Ok(())
}
